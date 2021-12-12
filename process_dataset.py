# import libararies
from PIL import Image
import xml.etree.ElementTree as ET
import os


def cropAndSave(image, root, category, coordinates, counter):
    im1 = image.crop(coordinates)
    filepath = image.filename.split("original dataset")[0]
    newname = root.find("./filename").text.split(".")[0] + "_" + str(counter) + ".png"
    path = os.path.join(filepath, f"Processed Dataset\{category}")
    os.makedirs(path, exist_ok=True)
    # width, height = im1.size
    # if (width > 50 and height > 50):
    im1.save(f"{path}\{newname}")


def nameFromXml(xmlName, workingDirect):
    tree = ET.parse(xmlName)
    root = tree.getroot()
    imageName = root.find("./filename").text
    # print(imageName)
    imageName = workingDirect + "\\original dataset\\images\\" + imageName
    return imageName, root


if __name__ == '__main__':
    workingDirect = os.getcwd()
    xmlNames = os.listdir(workingDirect + "\\original dataset\\annotations")
    xmlNames = [workingDirect + "\\original dataset\\annotations\\" + name for name in xmlNames]
    for xmlName in xmlNames:
        imageName, root = nameFromXml(xmlName, workingDirect)
        im = Image.open(imageName)
        # for all objects in the image
        counter = 0
        for obj in root.iter('object'):
            # determ properties of object
            for child in obj:
                if child.tag == 'name':
                    category = child.text
                if child.tag == 'bndbox':
                    coordinates = tuple(int(c.text) for c in child)
                    cropAndSave(im, root, category, coordinates, counter)
                    counter += 1

    width, height = im.size
    print(width, height)
