# 实例分割 = 目标检测 + 语义分割

BoundingBox[] bounding_boxes   # 使用BoundingBox 创建一个目标检测的锚框

uint8[] boxMask  # 存储Mask matrix data, size is (step * rows)