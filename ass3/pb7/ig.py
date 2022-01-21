from PIL import Image

def get_concat_h(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst

def get_concat_v(im1, im2):
    dst = Image.new('RGB', (im1.width, im1.height + im2.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height))
    return dst

im1 = Image.open('./falses/g9p7.png')
im2 = Image.open('./trues/g9p3.png')
im3 = Image.open('./trues/g9p9.png')

med = get_concat_h(im1, im2)
end = get_concat_h(med, im3)
end.save('res/9.png')