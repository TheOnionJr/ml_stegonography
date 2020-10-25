from stegano import lsb

secret = lsb.hide("E:\\ds-2\\train\\cover\\000001.jpg", "Hello World")
secret.save("kek.jpg")
