import bisect
import io
import json
import os
import pickle
import shutil
import struct
import tempfile
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from openpilot.common.file_chunker import get_existing_chunks, get_manifest_path

MODELS_DIR = Path(__file__).resolve().parent / 'models'
TG_INPUT_DEVICES_PATH = MODELS_DIR / 'tg_input_devices.json'
USBGPU_VID = 0xADD1
USBGPU_PID = 0x0001


def get_tg_input_devices(process_name: str, usbgpu: bool):
  with open(TG_INPUT_DEVICES_PATH) as f:
    return json.load(f)[process_name]['default' if not usbgpu else 'usbgpu']

def modeld_pkl_path(usbgpu: bool):
  prefix = 'big_' if usbgpu else ''
  return MODELS_DIR / f'{prefix}driving_tinygrad.pkl'

def dump_oob(obj, f):
  with tempfile.TemporaryFile(dir=".") as tmp:
    def buffer_callback(pb: pickle.PickleBuffer):
      m = pb.raw()
      tmp.write(struct.pack('<q', m.nbytes))
      tmp.write(m)
      pb.release() # keep peak ram at ~1 buffer
    stream = io.BytesIO()
    pickle.Pickler(stream, protocol=5, buffer_callback=buffer_callback).dump(obj)
    opcodes = stream.getvalue()
    f.write(struct.pack('<q', len(opcodes)))
    f.write(opcodes)
    tmp.seek(0)
    shutil.copyfileobj(tmp, f)

def load_oob(f):
  opcodes = f.read(struct.unpack('<q', f.read(8))[0])
  def buffers():
    prev = None
    while (h := f.read(8)):
      if prev is not None:
        prev.release()
      buf = bytearray(struct.unpack('<q', h)[0])
      f.readinto(buf)
      prev = pickle.PickleBuffer(buf)
      yield prev
  return pickle.load(io.BytesIO(opcodes), buffers=buffers())

@dataclass
class BufferSource:
  offset: int
  size: int
  dest_offset: int

def load_oob_stream(path):
  from tinygrad.device import Buffer, Device
  from tinygrad.dtype import dtypes

  paths = get_existing_chunks(path)
  if paths[0] == get_manifest_path(path):
    paths = paths[1:]
  starts = [0]
  for file_path in paths:
    starts.append(starts[-1] + os.path.getsize(file_path))

  def file_chunks(offset, size, max_size):
    while size:
      index = bisect.bisect_right(starts, offset) - 1
      file_offset = offset - starts[index]
      chunk_size = min(size, max_size, starts[index + 1] - offset)
      yield paths[index], file_offset, chunk_size
      offset += chunk_size
      size -= chunk_size

  def read_at(offset, size):
    data = bytearray(size)
    data_offset = 0
    for file_path, file_offset, chunk_size in file_chunks(offset, size, size):
      with open(file_path, 'rb') as file:
        read = os.preadv(file.fileno(), [memoryview(data)[data_offset:data_offset + chunk_size]], file_offset)
      assert read == chunk_size
      data_offset += chunk_size
    return data

  opcode_size = struct.unpack('<q', read_at(0, 8))[0]
  opcodes = read_at(8, opcode_size)
  with ThreadPoolExecutor(max_workers=1) as executor:
    device = executor.submit(Device.__getitem__, 'AMD')
    sources = []
    dest_offset = 0
    offset = 8 + opcode_size
    while offset < starts[-1]:
      size = struct.unpack('<q', read_at(offset, 8))[0]
      sources.append(BufferSource(offset + 8, size, dest_offset))
      dest_offset = (dest_offset + size + 0xfff) & -0x1000
      offset += 8 + size
    amd = device.result()

  storage = Buffer('AMD', dest_offset, dtypes.uint8).allocate()
  pending = deque(sources)
  regular = []

  def buffers():
    for _ in sources:
      yield memoryview(b'\0')

  def load_buffer(device, size, dtype, opaque=None, options=None, initial_value=None,
                  uop_refcount=0, base=None, offset=0, preallocate=False):
    if initial_value is not None:
      source = pending.popleft()
      assert source.size == size * dtype.itemsize
      if device == 'AMD':
        return Buffer(device, size, dtype, None, options, None, uop_refcount, storage, source.dest_offset, True)
      buffer = Buffer(device, size, dtype, opaque, options, None, uop_refcount, base, offset, preallocate)
      buffer.allocate()
      regular.append((buffer, source))
      return buffer
    return Buffer(device, size, dtype, opaque, options, None, uop_refcount, base, offset, preallocate)

  class ModelUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
      if module == 'tinygrad.device' and name == 'Buffer':
        return load_buffer
      return super().find_class(module, name)

  with ThreadPoolExecutor(max_workers=1) as executor:
    model = executor.submit(ModelUnpickler(io.BytesIO(opcodes), buffers=buffers()).load)
    chunk_size = amd.allocator.b[0].size
    chunks = []
    for source in sources:
      offset = 0
      for file_path, file_offset, size in file_chunks(source.offset, source.size, chunk_size):
        chunks.append((storage._buf.offset(source.dest_offset + offset), file_path, file_offset, size))
        offset += size
    def stream():
      buffers = [bytearray(chunk_size) for _ in range(2)]
      files = {}

      def read_chunk(buffer, chunk):
        _, path, offset, size = chunk
        if path not in files:
          files[path] = os.open(path, os.O_RDONLY)
        data = memoryview(buffer)[:size]
        read = os.preadv(files[path], [data], offset)
        assert read == size
        return data

      try:
        iterator = iter(chunks)
        first = next(iterator)
        with ThreadPoolExecutor(max_workers=1) as reader:
          future = reader.submit(read_chunk, buffers[0], first)
          for i, chunk in enumerate(iterator, 1):
            data = future.result()
            future = reader.submit(read_chunk, buffers[i % 2], chunk)
            try:
              yield data
            finally:
              data.release()
          data = future.result()
          try:
            yield data
          finally:
            data.release()
      finally:
        for fd in files.values():
          os.close(fd)

    data = stream()
    try:
      amd.allocator._copyin_stream([(dest, size) for dest, _, _, size in chunks], data)
    finally:
      data.close()
    model = model.result()

  assert not pending
  for buffer, source in regular:
    buffer.copyin(memoryview(read_at(source.offset, source.size)))

  return model

def usbgpu_present() -> bool:
  for d in Path("/sys/bus/usb/devices").glob("*"):
    try:
      if int((d / "idVendor").read_text(), 16) == USBGPU_VID and \
          int((d / "idProduct").read_text(), 16) == USBGPU_PID:
        return True
    except Exception:
      pass
  return False
