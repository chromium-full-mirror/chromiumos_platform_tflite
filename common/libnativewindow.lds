VERS_1.0 {
  # Export only AHardwareBuffer_* symbols in libnativewindow.so.
  global:
    AHardwareBuffer_*;

  # Hide everything else.
  local:
    *;
};
