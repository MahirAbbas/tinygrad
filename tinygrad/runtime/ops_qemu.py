from typing import Any, Optional
import ctypes, subprocess, pathlib, tempfile
from tinygrad.device import Compiled, Compiler, Allocator, BufferOptions
from tinygrad.renderer.cstyle import ClangRenderer

class QEMUCompiler(Compiler):
  def __init__(self):
    print("QEMU Initialised")
    super().__init__(cachekey="qemu")
    
  def compile(self, src:str) -> bytes:
    raise NotImplementedError("QEMU compilation not implemented")
  
class QEMUProgram:
  def __init__(self):
    pass


class QEMUAllocator(Allocator):
  def _alloc(self, size:int, options:BufferOptions): raise NotImplementedError("need alloc")
  def _free(self, opaque, options:BufferOptions): pass
  def copyin(self, dest, src:memoryview): raise NotImplementedError("need copyin")
  def copyout(self, dest:memoryview, src): raise NotImplementedError("need copyout")
  
  
class QEMUDevice(Compiled):
  def __init__(self, device:str):
    self.device = None
    # runtime = QEMURuntime()
    super().__init__(device, QEMUAllocator(self), ClangRenderer(), QEMUCompiler)
    
  def synchronize(self):
    pass
  

  