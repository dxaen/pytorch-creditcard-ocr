import sys
import cv2
import argparse
from ocr import card_has_number
import logging
from create_ocr import create_mobilenetv2_ocr, create_mobilenetv2_ocr_predictor

def parse_args():
    parser = argparse.ArgumentParser(description='OCR demo video capture')
    parser.add_argument('--model_path', default = './checkpoint/mb2-e3-vloss-5-98.pth')
    parser.add_argument('--label_path', default= './checkpoint/ocr_labels.txt')
    parser.add_argument('--video_file', default = None)
    parser.add_argument('--save_final_frame', default = './final_frame.png')
    return parser.parse_args()

def run_demo(args):
    if args.video_file:
        cap = cv2.VideoCapture(args.video_file)  # capture from file
    else:
        cap = cv2.VideoCapture(0)   # capture from camera
        cap.set(3, 1920)
        cap.set(4, 1080)

    class_names = [name.strip() for name in open(args.label_path).readlines()]
    net = create_mobilenetv2_ocr(len(class_names),width_mult=0.5, is_test=True)
    net.load(args.model_path)
    predictor = create_mobilenetv2_ocr_predictor(net, candidate_size=200)

    while True:
        ret, orig_image = cap.read()
        if orig_image is None:
            continue
        image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
        boxes, labels, probs = predictor.predict(image, 20, 0.50)
        for i in range(boxes.size(0)):
            box = boxes[i, :]
            label = f"{class_names[labels[i]]}: {probs[i]:.2f}"
            cv2.rectangle(orig_image, (int(box[0]), int(box[1])), 
                    (int(box[2]), int(box[3])), (255, 255, 0), 4)

            cv2.putText(orig_image, label,
                        (int(box[0])+20, int(box[1])+40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,  # font scale
                        (255, 0, 255),
                        2)  # line type
            
        cv2.imshow('annotated', orig_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        condition, number = card_has_number(boxes.numpy(), labels.numpy(), probs.numpy())
        if condition:
            logging.info(f"Number is :{number}")
            cv2.imwrite(args.save_final_frame, orig_image)
            break 
        else:
            logging.debug(f"Partial Number {number}")
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format='\n%(message)s')

    args = parse_args()
    run_demo(args)
