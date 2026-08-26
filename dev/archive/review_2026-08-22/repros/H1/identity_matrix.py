"""H1: systematic eq/hash matrix over every value-hashed / aux object.
For each: (a) rebuilt-identical -> equal+same hash; (b) genuinely different -> unequal;
(c) subclass -> unequal; (d) dtype-only / spelling-only differences; (e) mutation after first hash."""
import dataclasses as dc
import numpy as np
import jax, jax.numpy as jnp
import t3toolbox.tucker_tensor_train as t3
import t3toolbox.uniform_tucker_tensor_train as ut3
import t3toolbox.manifold as t3m
import t3toolbox.uniform_manifold as um
import t3toolbox.shared_geometry as sg
from t3toolbox.backend import geometry as bgeo, fitting as bfit, uniform_fitting as ufit
from t3toolbox.backend.common import partition_static, StaticSkeleton, ValueHashedFields

def chk(label, cond):
    print(('OK   ' if cond else 'FAIL ') + label)

np.random.seed(0)
SH, TK, TT = (6, 6, 4), (2, 3, 2), (1, 2, 2, 1)
x = t3.TuckerTensorTrain.randn(SH, TK, TT)
ux = ut3.UniformTuckerTensorTrain.from_t3(x)

print('--- ragged GeometryOps')
a, b = bgeo.ManifoldGeometryOps(), bgeo.ManifoldGeometryOps()
chk('rebuilt manifold ops equal', a == b and hash(a) == hash(b))
chk('manifold != corewise', a != bgeo.CorewiseGeometryOps())
chk('with_sharing derived != base', a.with_sharing((0, 0, 1), SH) != a)
chk('with_sharing rebuilt equal', a.with_sharing((0, 0, 1), SH) == b.with_sharing((0, 0, 1), SH))
chk('sharing (0,0,1) vs labels (7,7,3) same partition -> equal', a.with_sharing((0, 0, 1), SH) == a.with_sharing((7, 7, 3), SH))
chk('sharing None == all-singleton', a.with_sharing(None, SH) == a.with_sharing((0, 1, 2), SH) == a)
@dc.dataclass(frozen=True, eq=False)
class SubOps(bgeo.ManifoldGeometryOps):
    pass
chk('subclass (no new fields) != base', SubOps() != a and hash(SubOps()) != hash(a))

print('--- uniform GeometryOps')
ua, ub = bgeo.UniformManifoldGeometryOps.from_point(ux.data), bgeo.UniformManifoldGeometryOps.from_point(ux.data)
chk('rebuilt uniform manifold ops equal', ua == ub and hash(ua) == hash(ub))
ux2 = ut3.UniformTuckerTensorTrain.from_t3(t3.TuckerTensorTrain.randn(SH, (2, 2, 2), TT))
chk('different rank -> unequal', ua != bgeo.UniformManifoldGeometryOps.from_point(ux2.data))
# dtype-only: int8 masks with identical content
m_int8 = tuple(m.astype(np.int8) for m in ux.masks.data)
u_int8 = bgeo.UniformCorewiseGeometryOps(tuple(SH), m_int8)
u_bool = bgeo.UniformCorewiseGeometryOps(tuple(SH), tuple(ux.masks.data))
chk('int8 vs bool masks -> UNEQUAL keys (recompile, not wrong)', u_int8 != u_bool)
# mutation after first hash -> stale cached key
masks_copy = tuple(m.copy() for m in ux.masks.data)
um1 = bgeo.UniformCorewiseGeometryOps(tuple(SH), masks_copy)
h0 = hash(um1)
masks_copy[0][...] = False           # mutate the array the geometry holds, AFTER its key was cached
um2 = bgeo.UniformCorewiseGeometryOps(tuple(SH), tuple(m.copy() for m in masks_copy))
print('     mutated-after-hash: hash unchanged =', hash(um1) == h0,
      '; equals a fresh object with the MUTATED content =', um1 == um2,
      '; equals a fresh object with the ORIGINAL content =', um1 == u_bool)
chk('stored masks are writeable (not frozen via setflags)', ux.masks.data[0].flags.writeable)

print('--- SamplingKind hierarchy')
chk('APPLY == ApplyKind()', bfit.APPLY == bfit.ApplyKind() and hash(bfit.APPLY) == hash(bfit.ApplyKind()))
chk('ApplyKind != EntriesKind', bfit.ApplyKind() != bfit.EntriesKind())
k1 = bfit.probe_derivatives_kind(2, [1.0, 0.5, 0.25])
k2 = bfit.probe_derivatives_kind(2, np.array([1.0, 0.5, 0.25]))
chk('weight list vs ndarray -> equal', k1 == k2 and hash(k1) == hash(k2))
k3 = bfit.probe_derivatives_kind(2, np.array([1.0, 0.5, 0.25], dtype=np.float32))
chk('weight float32 input -> canonicalized to float64 -> equal', k1 == k3)
k4 = bfit.probe_derivatives_kind(2, jnp.array([1.0, 0.5, 0.25]))
chk('weight jax array input -> numpy canonical -> equal', k1 == k4 and isinstance(k4.weight, np.ndarray))
chk('order 1 vs 2 unequal', bfit.probe_derivatives_kind(1) != bfit.probe_derivatives_kind(2))
chk('order np.int64(2) == 2', bfit.probe_derivatives_kind(np.int64(2)) == bfit.probe_derivatives_kind(2))
chk('chunk_size None vs 100 unequal', bfit.probe_derivatives_kind(2, chunk_size=None) != bfit.probe_derivatives_kind(2))
chk('weight None vs explicit ones unequal (recompile only)', bfit.probe_derivatives_kind(2) != bfit.probe_derivatives_kind(2, [1, 1, 1]))
chk('dc.replace(chunk_size=) gives a distinct key', dc.replace(k1, chunk_size=None) != k1)
try:
    dc.replace(bfit.APPLY, forward=lambda *a: 0)
    chk('dc.replace(forward=) raises', False)
except TypeError as e:
    chk('dc.replace(forward=) raises TypeError', True)
# uniform kinds
uk1 = ufit.UniformProbeDerivativesKind.from_point(ux.data, order=2, weight=[1, 0.5, 0.25])
uk2 = ufit.UniformProbeDerivativesKind.from_point(ux.data, order=2, weight=[1, 0.5, 0.25])
chk('uniform kind rebuilt equal', uk1 == uk2 and hash(uk1) == hash(uk2))
chk('uniform kind != ragged kind of same params', uk1 != k1 and k1 != uk1)
chk('uniform kind at other rank unequal', uk1 != ufit.UniformProbeDerivativesKind.from_point(ux2.data, order=2, weight=[1, 0.5, 0.25]))
print('     uniform field order:', [f.name for f in dc.fields(uk1)])
# stray-attribute guard
@dc.dataclass(frozen=True, eq=False)
class Stray(bfit.ApplyKind):
    def __post_init__(self):
        object.__setattr__(self, 'scale', 2.0)
try:
    hash(Stray()); chk('stray attribute guard fires', False)
except TypeError:
    chk('stray attribute guard fires', True)
# jax-array-valued field (masks as jax arrays) -> hashable?
jm = tuple(jnp.asarray(m) for m in ux.masks.data)
try:
    hash(ufit.UniformApplyKind(shape=tuple(SH), masks=jm)); print('     jax masks in a kind field: hash OK')
except TypeError as e:
    print('     jax masks in a kind field: hash raises TypeError:', str(e)[:80])

print('--- mask holders')
M1, M2 = ut3.UT3Masks(*ux.masks.data), ut3.UT3Masks(*(m.copy() for m in ux.masks.data))
chk('UT3Masks rebuilt equal', M1 == M2 and hash(M1) == hash(M2))
Mi = ut3.UT3Masks(*(m.astype(np.int8) for m in ux.masks.data))
print('     UT3Masks bool vs int8 same content: eq =', M1 == Mi, ', hash equal =', hash(M1) == hash(Mi))
Mj = ut3.UT3Masks(*(jnp.asarray(m) for m in ux.masks.data))
try:
    print('     UT3Masks with jax masks: hash OK =', isinstance(hash(Mj), int), '; eq with numpy twin =', M1 == Mj)
except Exception as e:
    print('     UT3Masks with jax masks raises:', type(e).__name__, str(e)[:80])
# a 0-d mask / mismatched-shape content
chk('UT3Masks different shape unequal', M1 != ut3.UT3Masks(ux.masks.data[0][:, :1], ux.masks.data[1]))

print('--- SharedGeometry')
s1, s2 = sg.shared_manifold((0, 0, 1)), sg.shared_manifold([0, 0, 1])
chk('shared rebuilt equal', s1 == s2 and hash(s1) == hash(s2))
chk('shared manifold != shared corewise', s1 != sg.shared_corewise((0, 0, 1)))
chk('shared ragged != shared uniform', s1 != sg.shared(um.UNIFORM_MANIFOLD, (0, 0, 1)))
class SubShared(sg.SharedGeometry):
    pass
chk('SharedGeometry subclass unequal', SubShared(t3m.MANIFOLD, (0, 0, 1)) != s1 and hash(SubShared(t3m.MANIFOLD, (0, 0, 1))) != hash(s1))
chk('labels (0,0,1) vs (False,False,True) -> equal (same partition)', s1 == sg.shared_manifold((False, False, True)))

print('--- StaticSkeleton / partition_static')
fr = bgeo.UniformManifoldGeometryOps.from_point(ux.data).frame((ux.tucker_supercore, ux.tt_supercore))
d1, sk1 = partition_static(fr); d2, sk2 = partition_static(fr)
chk('skeleton rebuilt equal', sk1 == sk2 and hash(sk1) == hash(sk2))
chk('skeleton: masks kept static (no bool leaves)', not any(getattr(l, 'dtype', None) == np.bool_ for l in d1))
print('     static leaves of a uniform frame:', [t for t in str(sk1.tree).split("'") if t in ('static',)].__len__(), 'static entries')
# np.int64 shape entries: are they static or dynamic?
fr_np = fr[:4] + (tuple(np.int64(s) for s in fr[4]),) + (fr[5],)
d3, sk3 = partition_static(fr_np)
print('     shape as np.int64 entries -> dynamic leaves:', len(d3), 'vs Python ints:', len(d1))
