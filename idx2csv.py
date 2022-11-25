def convert(image_idx, label_idx, output_csv, n):
    with open(image_idx, "rb") as input_image, open(label_idx, "rb") as input_label, open(output_csv, "w") as output:
        input_image.read(16)
        input_label.read(8)
        image_arr = []
        for i in range(n):
            image = [ord(input_label.read(1))]
            for j in range(28 * 28):
                image.append(ord(input_image.read(1)))
            image_arr.append(image)
        for image in image_arr:
            output.write(",".join(str(pix) for pix in image)+"\n")

convert("./data/idx/train-images.idx3-ubyte", "./data/idx/train-labels.idx1-ubyte", "./data/csv/train.csv", 60000)
convert("./data/idx/t10k-images.idx3-ubyte", "./data/idx/t10k-labels.idx1-ubyte", "./data/csv/test.csv", 10000)