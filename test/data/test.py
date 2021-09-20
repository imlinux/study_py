

def load():
    BATCH_SIZE = 10000
    img_list = []
    label_list = []
    with open("/home/dong/tmp/mnist/train.txt") as f:
        for i in f:
            img, label = i.replace("\n", "").split(" ");
            img_list.append(img)
            label_list.append(label)

            if len(img_list) == BATCH_SIZE:

                yield img_list, label_list

                img_list = []
                label_list = []

        if len(img_list) !=0:
            yield img_list, label_list

for batch_id, (img_list, label_list) in enumerate(load()):

    print(batch_id, len(img_list), len(label_list))