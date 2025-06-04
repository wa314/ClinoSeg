from ultralytics import YOLO


model = YOLO('yolov8m-seg.pt')  # load a pretrained model (recommended for training)
results = model.train(data='config.yaml',
                      epochs= 500, imgsz= 640, batch= 32, lr0= 0.0001, lrf= 0.1, patience= 20,
                      freeze=22, degrees=0.2, translate=0.1, shear=0.1, fliplr=0.5)