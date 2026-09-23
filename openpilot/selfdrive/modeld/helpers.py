import io
import pickle
import struct
import sys
from pathlib import Path

from openpilot.common.hardware import AGNOS
from openpilot.common.hardware.usb import CHESTNUT_USB_PRODUCT, USB_DEVICES_PATH, is_chestnut_usb_id

MODELS_DIR = Path(__file__).resolve().parent / 'models'


def modeld_pkl_path(chestnut: bool):
  prefix = 'big_' if chestnut else ''
  return MODELS_DIR / f'{prefix}driving_tinygrad.pkl'

def load_pickle_streamed(path):
  from tinygrad import Device, dtypes
  from tinygrad.device import Buffer
  with open(path, 'rb') as f:
    opcodes = f.read(struct.unpack('<q', f.read(8))[0])
    arena_size = Path(path).stat().st_size - f.tell()
    arena = Buffer(Device.DEFAULT, arena_size, dtypes.uchar, preallocate=True)
    chunk_size = 32 << 20
    chunk = bytearray(min(chunk_size, arena_size))
    staging = Buffer('PYTHON', len(chunk), dtypes.uchar, opaque=memoryview(chunk))
    for off in range(0, arena_size, chunk_size):
      size = min(chunk_size, arena_size - off)
      if f.readinto(memoryview(chunk)[:size]) != size:
        raise EOFError('truncated out-of-band pickle')
      arena.view(size, dtypes.uchar, off).ensure_allocated().copy_from(staging.view(size, dtypes.uchar, 0).ensure_allocated())
      Device[Device.DEFAULT].synchronize()

  def persistent_load(pid):
    return arena.view(*pid)

  unpickler = pickle.Unpickler(io.BytesIO(opcodes))
  unpickler.persistent_load = persistent_load
  return unpickler.load()


def load_oob(path, chestnut=False):
  from tinygrad import Context
  device = 'USB+AMD:LLVM' if chestnut else 'QCOM' if AGNOS else 'METAL' if sys.platform == 'darwin' else 'CPU:LLVM'
  with Context(DEV=device):
    if chestnut:
      return load_pickle_streamed(path)
    from tinygrad_repo.examples.openpilot.helpers import load_pickle
    return load_pickle(path, out_of_band=True)

def chestnut_present() -> bool:
  for d in USB_DEVICES_PATH.glob("*"):
    try:
      usb_id = (int((d / "idVendor").read_text(), 16), int((d / "idProduct").read_text(), 16))
      product = (d / "product").read_text().strip()
      if is_chestnut_usb_id(*usb_id) and product == CHESTNUT_USB_PRODUCT:
        return True
    except Exception:
      pass
  return False

def chestnut_compiled() -> bool:
  path = modeld_pkl_path(chestnut=True)
  return path.is_file() and all(
    (MODELS_DIR / f'big_driving_warp_{size}_tinygrad.pkl').is_file() for size in ('1344x760', '1928x1208'))
