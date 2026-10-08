"""Precise mesh bisection along a surface-projected seam path, by walking
face-to-face along the path and inserting real topology exactly where it
crosses each face's boundary - the same fundamental technique used by
Blender's own Knife tool and by prior-art mesh-cutting addons (e.g.
patmo141/cut_mesh's "Polytrim": trace the path by finding, within the
*current, already-known* face, which of its own edges the next segment
exits through, insert a vertex there, and step into the neighboring face
across that edge).

Deliberately reuses the *same* resampled + surface-projected point
sequence that geometry.curve_display builds the visible tube from (see
geometry.islands.resolve_curve_seam_edges), so the preview tube and the
actual cut are the same geometry by construction - they can't visually
disagree.

Two other designs were tried and abandoned before landing on this one
(see git history for the full account):

- Per-segment bmesh.ops.bisect_plane calls, closing the gaps they leave at
  shared endpoints with various proximity-based vertex-welding schemes.
  Worked on simple fixtures, broke down on a real, densely branching seam
  network - welding by proximity can't reliably tell "same intended
  point" apart from "two nearby but distinct seam lines."
- Extruding the path into a "ribbon" (a strip of triangles following each
  point's local surface normal) and cutting it into the mesh with
  bpy.ops.mesh.intersect(solver='EXACT'), Blender's own boolean-family
  operator. This does produce a clean, connected cut - when the ribbon's
  "blade height" happens to suit the local geometry. But there's no one
  right height: too tall and the ribbon reaches across unrelated nearby
  surface (visually confirmed on a detailed mesh - the ribbon audibly
  "crinkles" across neighboring folds instead of staying local to the one
  seam it's meant to cut, and can produce multiple ambiguous crossings);
  too short and it doesn't reliably pierce the surface at all. No single
  height worked well across an entire real, unevenly-detailed mesh.

The current approach has no such tuning parameter at all: it only ever
asks purely topological, locally-scoped questions ("does the next segment
exit *this* face, and if so through which of its own edges") with no
notion of distance/height to get wrong.
"""

import math

import bmesh
from mathutils import Vector


class MeshCutError(Exception):
    """Raised when the walking cut can't proceed."""


# cut_path_into_mesh's default vertex-reuse tolerance during the walk.
MERGE_DIST = 1e-4

# How far a non-forced hint point is allowed to be from a pre-existing real
# vertex and still count as "landing on it" (see the main loop's hint-point
# branch). Wider than MERGE_DIST: a seam curve running along a Mirror-
# modifier weld line doesn't project exactly onto the weld's own vertices -
# the classic nearest-surface tie between the two mirrored halves (see
# curve_display.mirror_bias's docstring) leaves a small but real residual
# offset, confirmed on a real hood mesh to be consistently a few times
# MERGE_DIST (~1-1.5e-4) but nowhere near a real mesh edge's own length
# (~1e-2), so widening just this hint-matching check - not general
# insertion/merge decisions elsewhere - is safe from over-merging distinct
# nearby vertices while still catching the weld-line case.
HINT_MATCH_TOL = 5e-4


def check_oversized_faces(bm, segment_length, factor=4.0):
    """Returns the count of faces whose longest edge exceeds
    segment_length * factor. Cutting still works correctly regardless (the
    face-walk only ever advances one real face at a time, with no
    unbounded-cut failure mode) - but a face much bigger than the seam's
    own resample spacing means the mesh is too coarse to represent the
    seam's curvature well, producing a chunkier cut than the tube preview
    suggests. This is a quality warning, not a correctness requirement."""
    threshold = segment_length * factor
    count = 0
    for f in bm.faces:
        if max(e.calc_length() for e in f.edges) > threshold:
            count += 1
    return count


def _vertex_key(co, precision=6):
    return tuple(round(c, precision) for c in co)


def _on_face(face, p, tol=1e-4):
    """Originally used bmesh.geometry.intersect_face_point, guarded by an
    explicit plane-distance check, to test 2D containment - intersect_face_
    point only tests containment after projecting p onto the face's plane
    *along the face's own normal*, without checking how far p actually is
    from that plane, so a point on a completely different, unrelated face
    (e.g. a cube's top face) could register a false-positive "contains"
    match against a face whose plane it isn't anywhere near (e.g. a side
    face, if the point's projection along that face's normal happened to
    land within its silhouette) - confirmed experimentally: a point on a
    cube's top face falsely matched a side face this way, sending the whole
    walk off across the wrong face entirely.

    That plane-distance guard turned out not to be enough on its own: a
    small, perfectly ordinary triangle near one mesh boundary was later
    confirmed (on a Mirror-modifier weld seam fixture) to intersect_face_
    point-match target points literally a mesh-width away *and* lying near
    enough its plane to pass the guard too - the face's plane had a near-
    zero component along the very axis those far-away points varied in, so
    the plane-distance check alone couldn't tell them apart, and whatever
    intersect_face_point does internally for a point far outside a face's
    own small extent isn't reliably bounded to "no" either. Use
    _face_2d_margin instead - an explicit, real polygon-boundary distance
    computed in the face's own 2D basis - which has no such failure mode:
    a point genuinely far from a small face's footprint always ends up with
    a strongly negative margin, by construction."""
    if _face_2d_margin(face, p) < 0.0:
        return False
    return abs((p - face.verts[0].co).dot(face.normal)) <= tol


def _is_degenerate(face):
    """True if face has (near-)zero area - a collapsed/broken face (e.g.
    from an earlier insertion landing on top of an existing vertex by a
    hair's breadth) that must never be treated as a valid insertion target.
    A degenerate face's normal is the zero vector, which - left unguarded -
    makes every plane-distance check below trivially pass (dotting with a
    zero vector is always 0, "matching" every point in the entire mesh) and
    can silently vacuum up insertions from all over an unrelated seam into
    one spot: confirmed on a real, densely detailed mesh, one stray
    degenerate face did exactly this, fanning a single vertex into 70+
    bogus triangles bridging clear across the mesh - the "aren't aligned
    with any base-mesh face at all" artifact this guard exists to prevent."""
    return face.normal.length_squared < 1e-12


def _nearest_face_by_plane(faces, p, max_dist):
    """Fallback for when strict containment (_on_face) matches nothing: a
    point sitting exactly on (or extremely close to) a shared edge/vertex
    between faces can fail 2D polygon-boundary containment for *both*
    neighbors due to ordinary floating-point tie-breaking, even though
    it's clearly touching the mesh right there. Picking whichever
    candidate's plane the point is closest to (within max_dist) is a
    reasonable, safe fallback in that specific situation."""
    best = None
    for f in faces:
        if _is_degenerate(f):
            continue
        dist = abs((p - f.verts[0].co).dot(f.normal))
        if dist <= max_dist and (best is None or dist < best[0]):
            best = (dist, f)
    return best[1] if best is not None else None


def _face_2d_margin(face, p):
    """Signed distance (in the face's own plane) from p's projection to the
    face polygon's boundary: positive when strictly inside, ~0 on an edge,
    negative when outside. Unlike bmesh.geometry.intersect_face_point (which
    only gives a leaky boolean and false-positives a point *outside* a steep
    face whose silhouette it projects into - see _on_face), this is a real
    graded containment measure, so callers can pick the face a point is
    *most* inside of rather than just the first that loosely matches."""
    origin, u, v = _face_local_basis(face)
    px, py = _to_2d(origin, u, v, p)
    poly = [_to_2d(origin, u, v, fv.co) for fv in face.verts]
    n = len(poly)
    area2 = sum(
        poly[i][0] * poly[(i + 1) % n][1] - poly[(i + 1) % n][0] * poly[i][1]
        for i in range(n)
    )
    winding = 1.0 if area2 >= 0.0 else -1.0
    margin = float("inf")
    for i in range(n):
        ax, ay = poly[i]
        bx, by = poly[(i + 1) % n]
        ex, ey = bx - ax, by - ay
        edge_len = math.hypot(ex, ey)
        if edge_len < 1e-12:
            continue
        # signed perpendicular distance from p to this edge's line; the sign
        # is made "positive = interior side" using the polygon's own winding.
        cross = (ex * (py - ay) - ey * (px - ax)) / edge_len
        margin = min(margin, winding * cross)
    if margin == float("inf"):
        # Every edge was ~zero-length - a fully collapsed/degenerate face,
        # not merely "p is far outside a normal face". Must never look like
        # the *best* (most-inside) match to a full-mesh scan - see
        # _is_degenerate for why a naive +inf sentinel here is dangerous.
        return float("-inf")
    return margin


def _best_insertion_face(bm, p, tol=1e-4):
    """The face p should be inserted into: among all faces whose plane p is
    within tol of, the one p sits most deeply inside (largest 2D margin).
    Used only when forcing a genuine endpoint vertex and the local face-walk
    couldn't reach its home face (e.g. a steep Mirror-modifier weld/boundary
    face, where a plane-only nearest pick can hand back a face p is actually
    *outside* - poking that fans it into degenerate, non-severing triangles)."""
    best = None
    for f in bm.faces:
        if _is_degenerate(f):
            continue
        if abs((p - f.verts[0].co).dot(f.normal)) > tol:
            continue
        m = _face_2d_margin(f, p)
        if best is None or m > best[0]:
            best = (m, f)
    return best[1] if best is not None else None


def _resolve_insertion_face(bm, face, p, tol=1e-4):
    """face is where the walk currently believes a forced endpoint vertex p
    should be inserted (found via _on_face, whose 2D containment test can
    false-positive - see _on_face's own docstring - especially on a steep
    face, e.g. a Mirror-modifier weld/boundary face whose silhouette a
    nearby-but-outside point can still project into). Re-checked here with
    _face_2d_margin, a real graded containment measure: if p isn't actually
    inside face (margin meaningfully negative), redirect to whichever face
    it's genuinely most inside of instead. Poking a face at a point outside
    it fans it into degenerate triangles that never sever the boundary the
    point is really on (confirmed: this is exactly what let a transversal
    seam crossing a Mirror-modifier weld fail to separate its two sides)."""
    if _face_2d_margin(face, p) >= -tol:
        return face
    home = _best_insertion_face(bm, p, tol)
    return home if home is not None else face


def _face_containing_point(bm, p, tol=1e-4):
    """A full scan - only meant for bootstrapping the walk at the first
    point of a path; every subsequent step narrows the search to the
    small, local set of faces around the vertex just inserted.

    The plane-distance-only fallback (_nearest_face_by_plane, for when the
    strict _on_face scan matches nothing - see its own docstring for the
    boundary/tie-breaking case it's meant for) used to be the literal
    _nearest_face_by_plane call. That's unsafe as a *full-mesh* fallback
    specifically: with every face in the mesh as a candidate, it's no
    longer just breaking a tie between two small neighboring faces - it can
    just as easily match some completely unrelated, possibly degenerate
    face on the other side of the mesh whose plane merely happens to pass
    near p (confirmed on a real fixture: a stray sliver face spanning
    nearly the mesh's full length, with a plane nearly parallel to the
    query point's own path of travel, kept "nearest-by-plane" matching
    every point along a 1+ unit stretch of an unrelated seam). Use
    _best_insertion_face instead - same full-mesh plane-proximity
    candidate set, but picking whichever candidate p is most genuinely
    *inside*, not just planar-closest to."""
    hit = next((f for f in bm.faces if _on_face(f, p, tol)), None)
    if hit is not None:
        return hit
    return _best_insertion_face(bm, p, tol)


def _face_local_basis(face):
    normal = face.normal
    arbitrary = Vector((1.0, 0.0, 0.0)) if abs(normal.x) < 0.9 else Vector((0.0, 1.0, 0.0))
    u = normal.cross(arbitrary).normalized()
    v = normal.cross(u).normalized()
    return face.verts[0].co, u, v


def _to_2d(origin, u, v, p):
    d = p - origin
    return d.dot(u), d.dot(v)


def _segment_intersect_2d(p1, p2, p3, p4, eps=1e-9):
    """(t, point) where segment p1->p2 crosses segment p3->p4 (all 2D
    tuples), t the fraction along p1->p2, or None if they don't cross
    within both segments' bounds."""
    d1x, d1y = p2[0] - p1[0], p2[1] - p1[1]
    d2x, d2y = p4[0] - p3[0], p4[1] - p3[1]
    denom = d1x * d2y - d1y * d2x
    if abs(denom) < 1e-12:
        return None
    diffx, diffy = p3[0] - p1[0], p3[1] - p1[1]
    t = (diffx * d2y - diffy * d2x) / denom
    s = (diffx * d1y - diffy * d1x) / denom
    if -eps <= t <= 1 + eps and -eps <= s <= 1 + eps:
        return t, (p1[0] + d1x * t, p1[1] + d1y * t)
    return None


def _find_exit(face, p_from, p_to, skip_edge=None):
    """Within face's own boundary, finds which edge the segment
    (p_from -> p_to) exits through first, and the crossing point (3D) -
    or None if it doesn't exit this face (or is parallel/degenerate)."""
    origin, u, v = _face_local_basis(face)
    p1 = _to_2d(origin, u, v, p_from)
    p2 = _to_2d(origin, u, v, p_to)

    best = None
    for e in face.edges:
        if e is skip_edge:
            continue
        v1 = _to_2d(origin, u, v, e.verts[0].co)
        v2 = _to_2d(origin, u, v, e.verts[1].co)
        hit = _segment_intersect_2d(p1, p2, v1, v2)
        if hit is None:
            continue
        t, (x, y) = hit
        if t < 1e-7:
            continue  # crossing right at our own start point - not real progress
        if best is None or t < best[0]:
            world_pt = origin + u * x + v * y
            best = (t, e, world_pt)

    if best is None:
        return None
    return best[1], best[2]


def cut_path_into_mesh(bm, points, merge_dist=MERGE_DIST, vertex_cache=None):
    """Cuts bm's faces exactly along points (Vectors in bm's local space,
    already resampled and surface-projected - see geometry.curve_display),
    by walking face-to-face and inserting real vertices/edges exactly
    where the path crosses existing mesh topology. Mutates bm in place.
    Returns the set of new bmesh edges (BMEdge objects, not indices - a
    caller accumulating results across multiple calls against the same bm
    should keep these objects and only read .index, after
    bm.edges.ensure_lookup_table(), once every call is done, since indices
    go stale the moment anything else touches bm) forming the cut.

    Only points[0] and points[-1] are guaranteed to become real inserted
    vertices - everything in between is a direction/curvature *hint*, not
    a mandatory stop: if the walk can still reach a later point without
    crossing to a different face, no vertex is created for the points in
    between. This matters a lot in practice - a seam running across one
    large, flat face (no densify step to subdivide it first - see
    check_oversized_faces) still gets resampled into many closely-spaced
    points to hug curvature *elsewhere* along the seam, and forcing a real
    vertex at every single one of those - each requiring an interior
    "poke" (fan-triangulating the face around it, since it's nowhere near
    an edge) - was cascading into hundreds of ever-smaller sliver
    triangles for no reason: a straight run within one flat face needs
    exactly one edge, not one segment per resample point.

    vertex_cache, if given, is a dict (position-key -> BMVert) shared
    across multiple calls (e.g. one call per skeleton edge of a curve) so
    that a point shared between two calls - a branch/junction point in
    the original skeleton, or the closing point of a cyclic seam curve -
    reuses the exact same vertex both times, rather than each call
    creating its own coincident-but-separate one. Mutated in place; pass
    the same dict back in on the next call.
    """
    if vertex_cache is None:
        vertex_cache = {}
    if len(points) < 2:
        return set()

    def reuse(p):
        v = vertex_cache.get(_vertex_key(p))
        return v if (v is not None and v.is_valid) else None

    def find_existing_vertex(face, p, tol=merge_dist):
        """Like get_or_insert, but never creates new geometry - returns a
        real vertex already sitting within tol of p, or None if there
        isn't one. Used both as get_or_insert's own first fallback (at the
        default, tight tol=merge_dist), and - at a deliberately wider tol -
        by the main loop's hint-point branch below, to notice when a
        non-forced hint point still lands near a *pre-existing* real
        vertex - most commonly a seam curve running along an edge already
        shared between two faces, such as a Mirror-modifier weld line - so
        that already-real edge still gets connected/tagged as a seam
        boundary even though no new topology needs inserting there."""
        existing = reuse(p)
        if existing is not None:
            return existing
        # Search this face's own verts *and* its immediate neighbors' -
        # not just face.verts. A previous insertion (poke/edge_split) on
        # an adjacent, already-subdivided face can leave a real vertex
        # extremely close to p that isn't among face's own corners; without
        # checking those too, a second, independent nearby insertion
        # creates an overlapping duplicate instead of reusing it - which
        # in practice does happen and produces a genuine non-manifold
        # edge (confirmed: a real hood mesh had two faces sharing the same
        # 3 vertices with reversed winding right at one of these spots).
        nearby_faces = {face}
        for e in face.edges:
            nearby_faces.update(e.link_faces)
        for f in nearby_faces:
            for fv in f.verts:
                if (fv.co - p).length <= tol:
                    vertex_cache[_vertex_key(p)] = fv
                    return fv
        return None

    def get_or_insert(face, p):
        # HINT_MATCH_TOL, not merge_dist: an insertion landing within
        # merge_dist of an existing vertex was already the intended
        # reuse case find_existing_vertex's default tol covers, but a real
        # mesh consistently shows a few-times-merge_dist residual between
        # "this insertion's computed point" and "the real, structurally
        # significant vertex sitting right next to it" (same family as the
        # mirror-plane and boundary snapping above: two independent
        # computations that should coincide, differing by a little
        # floating-point/geometric noise) - confirmed: a real vertex just
        # 1.16x merge_dist from a poke()'d point was left with its own
        # untouched local connectivity fully intact, bypassing a seam that
        # otherwise cut cleanly right next to it. Widening only this
        # reuse check - not the edge-split distance check below, which is
        # about *where* to place a genuinely new vertex, a different
        # concern - stays local to `face`'s own immediate neighborhood
        # (nearby_faces, not a mesh-wide search), so it can't cause the
        # over-merging a blind mesh-wide weld at this tolerance would.
        existing = find_existing_vertex(face, p, tol=HINT_MATCH_TOL)
        if existing is not None:
            return existing
        for e in face.edges:
            v1, v2 = e.verts
            d = v2.co - v1.co
            denom = d.length_squared
            if denom < 1e-12:
                continue
            t = max(0.0, min(1.0, (p - v1.co).dot(d) / denom))
            closest = v1.co + d * t
            if (closest - p).length <= merge_dist:
                _new_edge, new_vert = bmesh.utils.edge_split(e, v1, t)
                new_vert.co = p
                vertex_cache[_vertex_key(p)] = new_vert
                return new_vert
        result = bmesh.ops.poke(bm, faces=[face])
        new_vert = result["verts"][0]
        new_vert.co = p
        vertex_cache[_vertex_key(p)] = new_vert
        return new_vert

    def ensure_edges(v1, v2):
        if v1 is v2:
            return []
        existing = next((e for e in v1.link_edges if v2 in e.verts), None)
        if existing is not None:
            return [existing]
        result = bmesh.ops.connect_vert_pair(bm, verts=[v1, v2])
        return list(result.get("edges", []))

    # Checked ahead of the expensive full-mesh _face_containing_point scan
    # below - profiled (on a real ~27-piece garment mesh) to be the single
    # dominant cost of the entire cut/flatten pipeline, >85% of total wall
    # time, almost all of it wasted: geometry.islands.resolve_curve_seam_edges
    # calls cut_path_into_mesh once per (post-modifier-evaluation) skeleton
    # edge, sharing one vertex_cache across all of them and guaranteeing
    # (via its own skeleton_point_cache) that a skeleton point shared
    # between two edges - the overwhelmingly common case for any connected
    # curve network, not just branch points - round-trips to the *exact*
    # same coordinate both times. So points[0] is already a real, cached
    # vertex on most calls; which of its faces we hand to get_or_insert
    # below doesn't matter in that case - find_existing_vertex checks this
    # exact same vertex_cache first, before ever looking at the face
    # argument at all - so this is a pure optimization, not a behavior
    # change: it only ever skips work that would have been thrown away.
    start_vertex = reuse(points[0])
    if start_vertex is not None:
        start_face = next(iter(start_vertex.link_faces), None)
    else:
        start_face = _face_containing_point(bm, points[0])
    if start_face is None:
        return set()

    current_vertex = get_or_insert(start_face, points[0])
    seam_edges = set()

    n = len(points)
    candidate_faces = list(current_vertex.link_faces)
    skip_edge = None

    for i in range(1, n):
        target = points[i]
        force = i == n - 1  # only the path's true final point must become a real vertex

        for _ in range(200):  # guards against a malformed/self-overlapping mesh looping forever
            if not candidate_faces:
                break
            contains_face = next((f for f in candidate_faces if _on_face(f, target)), None)
            if contains_face is not None:
                if force:
                    # _on_face's containment check can false-positive on a
                    # steep candidate face (see _resolve_insertion_face) -
                    # only actually matters once we're about to insert a
                    # real, permanent vertex here.
                    contains_face = _resolve_insertion_face(bm, contains_face, target)
                    target_vertex = get_or_insert(contains_face, target)
                    seam_edges.update(ensure_edges(current_vertex, target_vertex))
                    current_vertex = target_vertex
                    candidate_faces = list(current_vertex.link_faces)
                    skip_edge = None
                else:
                    # Reachable without crossing anything - ordinarily a
                    # true interior hint needing no vertex at all. But if
                    # this hint point happens to land exactly on a
                    # *pre-existing* real vertex distinct from
                    # current_vertex - most commonly because the seam
                    # curve is running along an edge already shared
                    # between two faces, e.g. a Mirror-modifier weld line
                    # - that real edge between them needs tagging as a
                    # seam boundary too, even though no new geometry needs
                    # inserting: "no vertex needed" only means "nothing to
                    # create", not "nothing to connect" (confirmed: a seam
                    # running exactly along a real hood mesh's mirror weld
                    # silently tagged *zero* of the weld's own edges as
                    # seam this way, leaving the two mirrored halves fully
                    # connected right through the "cut").
                    existing_vertex = find_existing_vertex(contains_face, target, tol=HINT_MATCH_TOL)
                    if existing_vertex is not None and existing_vertex is not current_vertex:
                        seam_edges.update(ensure_edges(current_vertex, existing_vertex))
                        current_vertex = existing_vertex
                        candidate_faces = list(current_vertex.link_faces)
                        skip_edge = None
                    # else: genuinely just a curvature hint mid-face - move
                    # on to the next (further) point as the new aim.
                break

            best = None
            for face in candidate_faces:
                hit = _find_exit(face, current_vertex.co, target, skip_edge)
                if hit is not None and best is None:
                    best = (face,) + hit
            if best is None:
                # Last resort before giving up on this leg: a point sitting
                # exactly on (or extremely close to) a shared edge/vertex
                # can fail strict boundary containment *and* look like it
                # doesn't exit any candidate face either, purely from
                # floating-point tie-breaking right at the boundary - not
                # because it's actually unreachable. Accept whichever
                # candidate face's plane it's closest to, if very close.
                fallback_face = _nearest_face_by_plane(candidate_faces, target, 1e-4)
                if fallback_face is None:
                    # Rare (confirmed on an irregular n-gon from a
                    # subsurf/geo-nodes stack): the 2D-in-face crossing
                    # test can miss a genuine exit on a non-planar or
                    # oddly-shaped face. Rather than abandon the leg
                    # entirely, fall back to a full-mesh search for
                    # whichever face genuinely contains the target and
                    # jump there directly - connect_vert_pair (already
                    # relied on elsewhere) robustly cuts a path through
                    # whatever's actually between the two points even
                    # when they're not on a shared/adjacent face.
                    # Same cache-first shortcut as the walk's own bootstrap
                    # above, and safe for the same reason (get_or_insert
                    # below checks vertex_cache before ever looking at the
                    # face) - this full-mesh fallback is rare in absolute
                    # terms but not free, and can still hit an already-
                    # cached point (e.g. re-approaching a skeleton vertex
                    # from an unexpected direction after a failed exit).
                    fallback_vertex = reuse(target)
                    fallback_face = (
                        next(iter(fallback_vertex.link_faces), None)
                        if fallback_vertex is not None
                        else _face_containing_point(bm, target)
                    )
                    if fallback_face is None:
                        break  # genuinely can't progress (e.g. an open mesh boundary) - drop this leg
                    target_vertex = get_or_insert(fallback_face, target)
                    seam_edges.update(ensure_edges(current_vertex, target_vertex))
                    current_vertex = target_vertex
                    candidate_faces = list(current_vertex.link_faces)
                    skip_edge = None
                    break
                if force:
                    # Only ever *insert* the forced endpoint into the face it
                    # genuinely sits inside, not merely one whose infinite
                    # plane it happens to be within 1e-4 of. On a Mirror-
                    # modifier weld/boundary the local candidate faces are
                    # steep (near-vertical, normal ~= +/-mirror axis), so a
                    # boundary endpoint well *outside* such a face in-plane can
                    # still be near its plane - _nearest_face_by_plane then
                    # hands it back, and get_or_insert poke()s that face at the
                    # outside point, fanning it into degenerate triangles that
                    # never sever the real boundary face the endpoint lies on
                    # (confirmed: a transversal seam crossing a mirror weld
                    # failed to separate its two halves this way). _on_face
                    # can't tell the difference - its 2D containment leaks a
                    # false positive on exactly these steep faces - so use a
                    # graded margin search for the endpoint's real home face.
                    home = _best_insertion_face(bm, target)
                    if home is not None:
                        fallback_face = home
                    target_vertex = get_or_insert(fallback_face, target)
                    seam_edges.update(ensure_edges(current_vertex, target_vertex))
                    current_vertex = target_vertex
                    candidate_faces = list(current_vertex.link_faces)
                    skip_edge = None
                break

            face, edge_hit, point_hit = best
            exit_vertex = get_or_insert(face, point_hit)
            seam_edges.update(ensure_edges(current_vertex, exit_vertex))
            current_vertex = exit_vertex
            skip_edge = edge_hit if edge_hit.is_valid else None
            candidate_faces = list(current_vertex.link_faces)

    return seam_edges
