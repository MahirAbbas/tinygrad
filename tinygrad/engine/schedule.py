import sys, atexit
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Callable, Set, Tuple, List, Dict, Optional, DefaultDict, cast
from tinygrad.ops import BUFFER_UOPS, UNSAFE_PAD_OPS, MetaOps, ReduceOps, UnaryOps, UOp, UOps, PatternMatcher, UPat, Variable, resolve, \
    graph_rewrite, track_rewrites, sint
from tinygrad.helpers import DEBUG, FUSE_CONV_BW, FUSE_ARANGE, Metadata, all_same, colored, diskcache_put, prod, dedup, all_int, merge_dicts, \
    getenv, unwrap
from tinygrad.dtype import ImageDType, dtypes
from tinygrad.shape.shapetracker import ShapeTracker
from tinygrad.shape.view import View, strides_for_shape
from tinygrad.engine.lazy import LazyBuffer
from tinygrad.device import Buffer

# creation can recurse a lot
sys.setrecursionlimit(10000)

BUF_LIMIT = {"METAL":32}
METAOPS = {MetaOps.COPY:UOps.COPY, MetaOps.EMPTY:UOps.EMPTY, MetaOps.VIEW:UOps.BUFFER_VIEW}

# *** ScheduleItem return type ***

@dataclass(frozen=True)
class ScheduleItem:
  ast: UOp
  bufs: Tuple[Buffer, ...]
  metadata: Tuple[Metadata, ...]
  @property
  def mutable_idxs(self) -> Tuple[int, ...]:
    """Writable buffer idxs"""
    return tuple(x.src[0].arg for x in (self.ast.src if self.ast.op is UOps.SINK else (self.ast,)))
  @property
  def outputs(self) -> Tuple[Buffer, ...]:
    """Read/write or write only buffers in the schedule."""
    return tuple(b for i,b in enumerate(self.bufs) if i in self.mutable_idxs)
  @property
  def inputs(self) -> Tuple[Buffer, ...]:
    """Read only buffers in the schedule."""
    return tuple(b for i,b in enumerate(self.bufs) if i not in self.mutable_idxs)

# *** UOp with VIEW (movementops) rewriting to UOp we can index ***

# ** helpers for doing movementops on uops

def st_fixup(u:UOp, apply_to_st:Callable[[ShapeTracker], ShapeTracker], cache:Dict[UOp, UOp]) -> UOp:
  if (n:=cache.get(u)) is not None: return n
  if u.op is UOps.VIEW: return u.replace(arg=apply_to_st(u.arg))
  if len(u.src) == 0 or (u.st is not None and u.st == apply_to_st(u.st)): return u
  cache[u] = ret = u.replace(src=tuple(st_fixup(x, apply_to_st, cache) for x in u.src))
  return ret

def permute_reduce(input_st:ShapeTracker, axis:Tuple[int, ...]) -> Tuple[ShapeTracker, Tuple[sint, ...]]:
  permute_axis = tuple(i for i in range(len(input_st.shape)) if i not in axis)+axis
  tmp = input_st.permute(permute_axis)
  return tmp, tmp.shape[-len(axis):]

# ** reduceop fusor

def view_r(view:UOp, r:UOp, rsrc:UOp) -> Optional[UOp]:
  if (st:=unwrap(view.st)).contiguous: return None
  tmp, rshape = permute_reduce(ShapeTracker.from_shape(unwrap(rsrc.st).shape), r.axis_arg)
  prshape = prod(rshape)
  strides = strides_for_shape(rshape)
  nv: List[View] = []
  for v in st.views:
    nv.append(View.create(v.shape+rshape, tuple(x*prshape for x in v.strides)+strides,
                          v.offset*prshape, v.mask+tuple((0,s) for s in rshape) if v.mask is not None else None))
  # update input_st and axis
  new_input_st = tmp + ShapeTracker(tuple(nv))
  _, new_rshape = permute_reduce(new_input_st, r.axis_arg)
  new_axis = tuple(range(len(new_input_st.shape)-len(new_rshape), len(new_input_st.shape)))
  return st_fixup(rsrc, lambda st:st+new_input_st, {}).r(r.arg[0], new_axis).view(ShapeTracker.from_shape(st.shape))

def push_swizzle_down_through_reduce(root:UOp, swizzle:UOp) -> UOp:
  swizzle_st, src_st = unwrap(swizzle.st), unwrap(swizzle.src[0].st)
  assert swizzle_st.contiguous, "can't push a non contiguous SWIZZLE down to STORE"
  assert prod(swizzle_st.shape) == prod(src_st.shape), "can't push expands down to STORE"
  output_shape = swizzle_st.reduce(root.axis_arg)
  new_axis = tuple(i for i,(s,u) in enumerate(zip(src_st.shape, output_shape)) if s != u)
  return swizzle.src[0].r(root.arg[0], new_axis).view(ShapeTracker.from_shape(output_shape))

def push_swizzle_down_through_elementwise(root:UOp) -> Optional[UOp]:
  swizzles = [x for x in root.src if x.op is UOps.VIEW and len(x.src) != 0]
  if len(swizzles) == 0: return None
  swizzle_shapes = [(unwrap(x.st).shape, unwrap(x.src[0].st).shape) for x in swizzles]
  assert all_same([(x, prod(x), prod(y)) for x,y in swizzle_shapes]), f"swizzles must have the same size {swizzle_shapes}"
  new_shape, new_input_shape = swizzle_shapes[0]
  fixup_cache: Dict[UOp, UOp] = {}
  new_srcs = [x.src[0] if x in swizzles else st_fixup(x, lambda st:st.reshape(new_input_shape), fixup_cache) for x in root.src]
  ret = UOp(root.op, root.dtype, tuple(new_srcs), root.arg)
  return ret if ret.op is UOps.STORE else ret.view(ShapeTracker.from_shape(new_shape))

def merge_double_reduce(root:UOp, first_reduce:UOp) -> UOp:
  assert root.arg[0] == first_reduce.arg[0], "can't merge reduceops with different alu"
  assert not any(x.op is UOps.REDUCE_AXIS for x in first_reduce.parents), "can't merge more than two reduceops at a time"
  return first_reduce.src[0].r(first_reduce.arg[0], root.axis_arg+first_reduce.axis_arg)

merge_views = PatternMatcher([(UPat(UOps.VIEW, src=(UPat(UOps.VIEW, name="s0"),), name="s1"), lambda s0,s1: s0.replace(arg=s0.st+s1.st))])

# push VIEW to loads
view_left = merge_views+PatternMatcher([
  # view before ALU
  (UPat(UOps.VIEW, src=(UPat((UOps.ALU, UOps.CAST, UOps.BITCAST, UOps.ASSIGN, UOps.CONTIGUOUS, *BUFFER_UOPS), name="e"),), name="v"),
   lambda e,v: e.replace(src=tuple(s.view(v.st) if s.has_st else s for s in e.src))),
])

# push VIEW to stores
view_right = merge_views+PatternMatcher([
  # ASSIGN can override st
  (UPat(UOps.STORE, src=(UPat.var("b"), UPat.var("st"), UPat(UOps.ASSIGN, name="a"))),
   lambda a,b,st: UOp.store(b, (a.arg[0]+st.arg).to_uop(), a.replace(arg=())) if a.arg else None),
  # view on reduce creates a new VIEW
  (UPat(UOps.VIEW, src=(UPat(UOps.REDUCE_AXIS, src=UPat.var("rsrc"), name="r"),), name="view"), view_r),
  # push a SWIZZLE down to STORE, through a reduce (ONLY reshapes)
  (UPat(UOps.REDUCE_AXIS, src=(UPat(UOps.VIEW, name="swizzle"),), name="root"), push_swizzle_down_through_reduce),
  # push SWIZZLE(s) down to STORE, through an elementwise op (ONLY reshapes)
  (UPat((UOps.ALU, UOps.CAST, UOps.BITCAST, UOps.ASSIGN, UOps.CONTIGUOUS, UOps.STORE), name="root"), push_swizzle_down_through_elementwise),
  (UPat(UOps.REDUCE_AXIS, src=(UPat(UOps.REDUCE_AXIS, name="first_reduce"),), name="root"), merge_double_reduce),
])

# ** ScheduleItem context builder

@dataclass(frozen=True)
class ScheduleItemContext:
  var_vals: Dict[Variable, int]
  sts: Set[ShapeTracker]
  bufs: List[UOp]
  preloads: List[UOp]
  outputs: Dict[UOp, UOp]
  assigned_to: Set[UOp]

def _append_st_vars(ctx:ScheduleItemContext, x:UOp) -> Optional[UOp]:
  if (st:=unwrap(x.st)) in ctx.sts: return None
  st, var_vals = st.simplify().unbind()
  ctx.var_vals.update(var_vals)
  ctx.sts.add(st)
  return st.to_uop() if st != x.st else None
append_st_vars = PatternMatcher([(UPat(UOps.VIEW, name="x"), _append_st_vars)])

def _append_buf(ctx:ScheduleItemContext, x:UOp) -> UOp:
  ctx.bufs.append(x)
  return UOp(UOps.DEFINE_GLOBAL, x.dtype, (), len(ctx.bufs)-1)
append_bufs = PatternMatcher([(UPat(UOps.BUFFER, name="x"), _append_buf),])

def _append_preload(ctx:ScheduleItemContext, b:UOp) -> None:
  if b in ctx.assigned_to and b not in ctx.outputs: ctx.preloads.append(b)
  return None
append_preloads = PatternMatcher([(UPat(UOps.LOAD, src=(UPat.var("b"), UPat()), arg=True), _append_preload),])

to_ast = PatternMatcher([
  (UPat(UOps.CONTIGUOUS, src=(UPat.var("x"),)), lambda x: x),
  (UPat(UOps.SINK, src=(UPat.store(UPat(), UPat(), UPat(tuple(METAOPS.values()), name="x")),)), lambda x: x),
])

fuse_multioutput = PatternMatcher([
  (UPat(UOps.LOAD, src=(UPat.var("b"), UPat()), arg=False), lambda ctx,b: ctx.outputs.get(b, None)),
])

PROCESS_REPLAY_CAPTURE: List[Tuple[UOp, UOp]] = []
if getenv("RUN_PROCESS_REPLAY"):
  @atexit.register
  def save_process_replay():
    for base_sink,ret in PROCESS_REPLAY_CAPTURE: diskcache_put("schedule_process_replay", str(base_sink.key), (base_sink, ret))

# *** LazyBuffer lowering to UOp ***

def to_uop(buf:LazyBuffer, buf_uops:Dict[Buffer, UOp], metadata:Dict[UOp, Metadata], cache:Dict[LazyBuffer, UOp]) -> UOp:
  if (r:=cache.get(buf)) is not None: return r
  # all movementops are UOps.VIEW
  if buf is not buf.base:
    cache[buf] = ret = to_uop(buf.base, buf_uops, metadata, cache).view(buf.st)
    return ret
  # consts are always fused and generated
  if buf.op is MetaOps.CONST: return buf_uops[buf.buffer]
  dtype = buf.dtype.base if isinstance(buf.dtype, ImageDType) else buf.dtype
  # preloads are always loaded
  if buf.is_realized(): return UOp(UOps.LOAD, dtype, (buf_uops[buf.buffer], buf.st.to_uop()), buf.is_realized())
  # otherwise we fuse it or STORE -> LOAD the value in global memory
  src = tuple(to_uop(x, buf_uops, metadata, cache) for x in buf.srcs)
  if buf.op in ReduceOps: ret = src[0].r(buf.op, buf.arg)
  elif buf.op is MetaOps.CONTIGUOUS: ret = UOp(UOps.CONTIGUOUS, dtype, src)
  elif buf.op is MetaOps.ASSIGN: ret = UOp(UOps.ASSIGN, dtype, (buf_uops[buf.buffer], src[1]), buf.arg)
  elif buf.op in METAOPS: ret = UOp(METAOPS[cast(MetaOps, buf.op)], buf.dtype, (buf_uops[buf.buffer], *src), buf.arg)
  elif buf.op is UnaryOps.CAST: ret = UOp(UOps.CAST, dtype, src)
  elif buf.op is UnaryOps.BITCAST: ret = UOp(UOps.BITCAST, dtype, src)
  else: ret = UOp(UOps.ALU, dtype, src, buf.op)
  cache[buf] = ret = ret if (ubuf:=buf_uops.get(buf.buffer)) is None else \
      UOp(UOps.LOAD, dtype, (ubuf, buf.st.to_uop(), UOp.store(ubuf, ShapeTracker.from_shape(buf.shape).to_uop(), ret)))
  if buf.metadata is not None: metadata[ret] = buf.metadata
  return ret

# *** DAG creation: decide which LazyBuffers should realize ***

def _recurse_lb(buf:LazyBuffer, realizes:Dict[LazyBuffer, None], allbufs:Dict[LazyBuffer, None], simple_pads:Dict[LazyBuffer, None],
                children:DefaultDict[LazyBuffer, Dict[LazyBuffer, None]], assign_targets:Dict[LazyBuffer, LazyBuffer],
                double_reduces:Dict[LazyBuffer, None], scheduled=False) -> None:
  """recursively search the entire graph for all LazyBuffers, insert realizes after expands"""
  if buf in allbufs: return None
  if buf.base.realized is not None: return realizes.setdefault(buf.base)
  # check if we need to realize views
  if buf is not buf.base:
    # fuse some pads
    if len(buf.st.views) == 1 and buf.st.views[-1].mask is not None and all_int(buf.base.st.shape) and \
        resolve(prod(buf.base.st.shape) >= prod([y-x for x,y in buf.st.views[-1].mask])):
      simple_pads[buf.base] = None
    # realize all expands
    elif resolve(prod(buf.base.st.shape) < prod(buf.st.shape)):
      # this was causing "test_lil_model" to fail
      if buf.base.op is UnaryOps.CAST and isinstance(buf.base.srcs[0].dtype, ImageDType) and isinstance(buf.base.arg, ImageDType):
        simple_pads[buf.base] = None # don't realize image to image casts. this is part of a larger problem
      else: realizes[buf.base] = None
    # check all other pads for safe fusion
    elif any(v.mask is not None for v in buf.st.views): simple_pads[buf.base] = None
    return _recurse_lb(buf.base, realizes, allbufs, simple_pads, children, assign_targets, double_reduces)
  if buf.op in ReduceOps and buf.srcs[0].base.op is buf.op and buf.srcs[0] is not buf.srcs[0].base: double_reduces[buf] = None
  allbufs[buf] = None
  if buf.forced_realize or buf.op in MetaOps: realizes[buf] = None
  if buf.op is MetaOps.ASSIGN:
    assign_targets[(target:=buf.srcs[0])] = buf
    assert target._base is None, f"assign must be to base {target}"
    assert target.is_realized(), f"assign must be already realized to schedule {target}"
  if buf.op is MetaOps.COPY:
    assert buf.srcs[0].st.contiguous and buf.srcs[0].size == buf.srcs[0].base.size, "can only copy contig"
    realizes[buf.srcs[0].base] = None
  if buf.op is MetaOps.VIEW: realizes[buf.srcs[0].base] = None
  for x in buf.srcs:
    if x.base.realized is None: children[x.base][buf] = None
    _recurse_lb(x, realizes, allbufs, simple_pads, children, assign_targets, double_reduces)

def _is_padding_okay(buf:LazyBuffer, realizes:Dict[LazyBuffer, None], cache:Dict[LazyBuffer, bool]) -> bool:
  if (n:=cache.get(buf)) is not None: return n
  if buf in realizes: return True
  # NOTE: this broke to_image_idx and coder with JIT
  if buf.op in UNSAFE_PAD_OPS: return False
  cache[buf] = ret = all(_is_padding_okay(x.base, realizes, cache) for x in buf.srcs)
  return ret

def _recursive_group(tr:LazyBuffer, st:ShapeTracker, r:LazyBuffer, children:DefaultDict[LazyBuffer, Dict[LazyBuffer, None]],
                     realizes:Dict[LazyBuffer, None], reduce_for_op:Dict[LazyBuffer, LazyBuffer], group:Dict[LazyBuffer, None],
                     cache:Dict[Tuple[LazyBuffer, ShapeTracker], None]) -> None:
  """recursively search the LazyBuffer for groupable children, realize the LazyBuffer if a child can't group"""
  if (tr, st) in cache: return
  cache.setdefault((tr, st))
  if tr in realizes and tr is not r:
    # can only fuse contiguous
    # max one reduceop per kernel
    if not st.contiguous or st.size != r.st.size or tr in reduce_for_op: group.setdefault(r)
    return group.setdefault(tr)
  for tr_next in children[tr]:
    # max one reduceop per kernel
    if tr_next.op in ReduceOps: return group.setdefault(r)
    # can only fuse contiguous
    if len(st_childs:=dedup(s for s in tr_next.srcs if s.base == tr)) > 1: return group.setdefault(r)
    _recursive_group(tr_next, st+st_childs[0].st, r, children, realizes, reduce_for_op, group, cache)

def _get_isolated_children(r:LazyBuffer, reduce_for_op:Dict[LazyBuffer, LazyBuffer], children:DefaultDict[LazyBuffer, Dict[LazyBuffer, None]],\
    realizes:Dict[LazyBuffer, None], group:Dict[LazyBuffer, None]) -> Dict[LazyBuffer, None]:
  rc_parents, cache = deque(group), set()
  while rc_parents:
    if (p:=rc_parents.pop()) in cache: continue
    cache.add(p)
    # max one reduceop per kernel
    if p.op in ReduceOps: return {}
    rc_parents.extend(x.base for x in p.srcs if x.base.realized is None and x.base is not r)
  # search descendants of the reduceop that can cleanly group
  descendants: Dict[LazyBuffer, None] = {}
  for tr in group: _recursive_group(tr, tr.st, tr, children, realizes, reduce_for_op, descendants, cache={})
  return merge_dicts([group, {} if any(tr in group for tr in descendants) else descendants])

def _append_store(stores:Dict[UOp, UOp], st:UOp, b:UOp, val:UOp, store:UOp) -> UOp:
  stores[b] = store
  return UOp(UOps.LOAD, val.dtype, (b, st), False)
append_stores = PatternMatcher([
  (UPat.load(UPat.var("b"), UPat.var("st"), UPat(UOps.STORE, src=(UPat.var("b"), UPat(), UPat.var("val")), name="store")), _append_store),
])

@track_rewrites(named=True)
def create_schedule_with_vars(outs:List[LazyBuffer]) -> Tuple[List[ScheduleItem], Dict[Variable, int]]:
  """create a graph for realizing the outputs"""
  # start by just realizing the buffers passed in
  realizes: Dict[LazyBuffer, None] = {x.base:None for x in outs if x.base.realized is None}
  allbufs: Dict[LazyBuffer, None] = {}
  simple_pads: Dict[LazyBuffer, None] = {}
  children: DefaultDict[LazyBuffer, Dict[LazyBuffer, None]] = defaultdict(dict)
  assign_targets: Dict[LazyBuffer, LazyBuffer] = {}
  double_reduces: Dict[LazyBuffer, None] = {}
  for out in outs: _recurse_lb(out.base, realizes, allbufs, simple_pads, children, assign_targets, double_reduces, scheduled=True)

  # check if we have to realize pads
  for p in simple_pads:
    if not _is_padding_okay(p, realizes, {}):
      realizes[p] = None

  # find all reduces, and pair them to a elementwise op. if they can't be cleanly paired, force realize the reduce (or a contig child)
  reduce_for_op: Dict[LazyBuffer, LazyBuffer] = {}
  reduce_of_const: List[LazyBuffer] = []
  for r in allbufs:
    if r.op not in ReduceOps or r in realizes: continue

    group: Dict[LazyBuffer, None] = {}
    _recursive_group(r, r.st, r, children, realizes, reduce_for_op, group, cache={})
    # max one reduceop per kernel
    can_chase = all(tr not in reduce_for_op for tr in group)
    # TODO: forced_realize exists because the scheduler is incapable of checking for self-contained DAGs
    forced_realize = r in group
    if not forced_realize and len(group) > 1:
      group = _get_isolated_children(r, reduce_for_op, children, realizes, group)
    # can only fuse assign if no other assign_target is used in the kernel
    if not forced_realize and any(x.op is MetaOps.ASSIGN for x in group):
      parents = deque((r, *group))
      while parents and not forced_realize:
        if (p:=parents.pop().base).realized or p in realizes:
          if p in assign_targets and assign_targets[p] not in group: forced_realize, can_chase = True, False
          continue
        parents.extend(p.srcs)
    if forced_realize or not group:
      tr = r
      if can_chase:
        # can chase this down to contiguous children
        st = tr.st
        while len(children[tr]) == 1:
          tr_next = next(iter(children[tr]))
          st_childs = dedup(s for s in tr_next.srcs if s.base is tr)
          if len(st_childs) > 1: break
          if st.size != st_childs[0].st.size: break
          st = st + st_childs[0].st
          if not st.contiguous or tr_next.op in ReduceOps: break
          tr = tr_next
        # don't cast to higher size before store (tr cannot be realized if forced_realize)
        if tr.op is UnaryOps.CAST and tr.arg.itemsize > tr.srcs[0].dtype.itemsize:
          tr = tr.srcs[0].base
        reduce_for_op[tr] = r
      realizes[tr] = None
    else: reduce_for_op.update((tr, r) for tr in group)
    if FUSE_ARANGE and r.op is ReduceOps.SUM and r.srcs[0].base.op is MetaOps.CONST: reduce_of_const.append(r)

  # fuse double reduces with no other child
  if FUSE_CONV_BW:
    for reduceop in double_reduces:
      top_reduce = reduceop.base.srcs[0].base
      if len(children[top_reduce]) == 1: del realizes[top_reduce]

  for r in reduce_of_const:
    group = {tr:None for tr,rop in reduce_for_op.items() if rop is r}
    if DEBUG_ARANGE:=(getenv("DEBUG_ARANGE")): print(f"checking {r} {group=}")
    if any(tr.forced_realize for tr in group) or any(x.base in group for x in outs): continue
    kernel_children = {c for tr in group for c in children[tr] if c.op not in {MetaOps.COPY, MetaOps.VIEW}}
    if len(kernel_children) == 0: continue
    if DEBUG_ARANGE: print(colored(f"folding {r}", "green"))
    for tr in group: del realizes[tr]

  output_groups: DefaultDict[LazyBuffer, List[UOp]] = defaultdict(list)
  buf_uops: Dict[Buffer, UOp] = {}
  uop_bufs: Dict[UOp, Buffer] = {}
  var_vals: Dict[Variable, int] = {}
  lazybufs_to_realize: Dict[Buffer, LazyBuffer] = {}
  assigned_to: Set[UOp] = set()
  for buf in realizes:
    if buf.realized is None and buf.op is not MetaOps.CONST:
      if (dup:=lazybufs_to_realize.get(buf.buffer)) is not None:
        raise RuntimeError(f"can't double realize in one schedule, Buffer is realizing both {dup} and {buf}")
      lazybufs_to_realize[buf.buffer] = buf

      # make things that can't be images not images
      if isinstance(buf.dtype, ImageDType) and (prod(buf.shape) != prod(buf.dtype.shape) or
                                                not any(buf.shape[x]%4 == 0 for x in buf.st.unit_stride_axes())):
        if DEBUG >= 2: print(f"forcing image {buf.dtype} with shape {buf.shape} to float32")
        buf.dtype = dtypes.float32
        # hack the underlying buffer too
        if buf.base is buf:
          assert not hasattr(buf.buffer, '_buf'), "can't fixup allocated buffer"
          buf.buffer.dtype = dtypes.float32
          buf.buffer.options = None
    if buf.op is MetaOps.CONST:
      if isinstance(val:=buf.arg, UOp): var_vals.update([val.unbind()])
      uop = UOp(UOps.VALID, dtypes.bool, (buf.st.to_uop(),)).where(v:=UOp.const(buf.dtype.scalar(), buf.arg), v.const_like(0))
    # NOTE: UOps.BUFFER creation must come after the ImageDType fixup
    else: uop = UOp(UOps.BUFFER, buf.buffer.dtype.ptr(), (), (len(buf_uops), (buf.buffer.device, buf.buffer.size, buf.buffer.dtype)))
    if buf.buffer not in buf_uops:
      buf_uops[buf.buffer] = uop
      uop_bufs[uop] = buf.buffer
    if buf.realized is None and buf.op is not MetaOps.CONST:
      if buf.op is MetaOps.ASSIGN: assigned_to.add(uop)
      output_groups[reduce_for_op.get(buf, buf)].append(buf_uops[buf.buffer])

  # this is the big graph
  metadata: Dict[UOp, Metadata] = {}
  cache: Dict[LazyBuffer, UOp] = {}
  sink = UOp(UOps.SINK, dtypes.void, tuple(to_uop(x, buf_uops, metadata, cache) for x in outs))
  stores: Dict[UOp, UOp] = {}
  graph_rewrite(sink, append_stores, stores)
  # break the big graph into ScheduleItems
  prescheduled: List[ScheduleItem] = []
  assign_preloads: List[List[UOp]] = []
  for outbufs in output_groups.values():
    sink = UOp(UOps.SINK, dtypes.void, tuple(stores[b] for b in outbufs))
    ctx = ScheduleItemContext(var_vals, set(), [], [], {x.src[0]:x.src[2] for x in sink.src}, assigned_to)
    # fuse multi output store -> loads
    if len(ctx.outputs) > 1: sink = graph_rewrite(sink, fuse_multioutput, ctx)
    # swizzling
    sink = graph_rewrite(graph_rewrite(sink, view_left), view_right)
    # add assign preloads in this schedule
    if len(ctx.assigned_to) != 0: sink = graph_rewrite(sink, append_preloads, ctx)
    # append bufs and var_vals
    sink = graph_rewrite(graph_rewrite(sink, to_ast), append_st_vars+append_bufs, ctx)
    prescheduled.append(si:=ScheduleItem(sink, tuple(uop_bufs[b] for b in ctx.bufs), ()))
    assign_preloads.append(ctx.preloads)
  schedule_targets = {out:lsi for lsi in prescheduled for out in lsi.outputs}

  graph: DefaultDict[ScheduleItem, List[ScheduleItem]] = defaultdict(list)
  in_degree: DefaultDict[ScheduleItem, int] = defaultdict(int)
  for i,lsi in enumerate(prescheduled):
    if lsi not in in_degree: in_degree[lsi] = 0
    # realize outputs before a parent is assigned to
    parents_assigns = dedup(xsi for x in assign_preloads[i] if (xsi:=schedule_targets.get(uop_bufs[x])))
    for assign in parents_assigns:
      graph[lsi].append(assign)
      in_degree[assign] += 1
    # realize outputs after all parents are realized
    scheduled_parents = dedup(xsi for x in lsi.inputs if (xsi:=schedule_targets.get(x)) is not None and xsi not in parents_assigns)
    for x in scheduled_parents:
      graph[x].append(lsi)
      in_degree[lsi] += 1

  queue = deque(lsi for lsi,deg in in_degree.items() if deg == 0)
  schedule: List[ScheduleItem] = []
  while queue:
    schedule.append(si:=queue.popleft())
    for b in si.outputs: del lazybufs_to_realize[b].srcs  # can only schedule once
    if (m:=BUF_LIMIT.get(device:=si.outputs[0].device)) and len(si.bufs) >= m:
      if DEBUG >= 3: print(si)
      raise RuntimeError(f"Kernel for {si.metadata} exceeded the {m} buffer count limit for {device} with {len(si.bufs)} buffers.")
    for x in graph[si]:
      in_degree[x] -= 1
      if in_degree[x] == 0: queue.append(x)

  # confirm everything was scheduled correctly
  if any(degree != 0 for degree in in_degree.values()) or len(in_degree) != len(schedule):
    raise RuntimeError(f"cycle detected in graph, prescheduled {len(in_degree)} but only scheduled {len(schedule)}")
  if DEBUG >= 1 and len(schedule) >= 10: print(f"scheduled {len(schedule)} kernels")
  return schedule, var_vals

def create_schedule(outs:List[LazyBuffer]) -> List[ScheduleItem]:
  schedule, var_vals = create_schedule_with_vars(outs)
  assert len(var_vals) == 0
  return schedule
