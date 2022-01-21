import base64
for i in range(16):
    f=open(str(i+1)+'.png','rb') #二进制方式打开图文件
    ls_f=base64.b64encode(f.read()) #读取文件内容，转换为base64编码

    f.close()
    print("[figure"+str(i+1)+"]:data:image/png;base64," + str(ls_f))
    print()