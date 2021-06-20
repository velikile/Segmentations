import os
import posixpath
import http.server
import urllib.request, urllib.parse, urllib.error
import html
import shutil
import mimetypes
import re
from io import BytesIO
import base64
from pathlib import Path
import json
import numpy as np 
import torch
from PIL import Image
from torchvision import transforms
import random as rnd

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model = torch.load("mrcnn.m")
model.eval()
rand_colors = np.random.randint(0,255,size=(1000,3))
def ProduceMask(fileBytes): 
    img = transforms.ToTensor()(Image.open(BytesIO(fileBytes)))
    with torch.no_grad():
        prediction = model([img.to(device)])
        imgmask   = Image.fromarray(prediction[0]['masks'][0, 0].mul(255).byte().cpu().numpy())
        maskcolor = np.stack((imgmask,)*3,axis=-1)
        for m in range(prediction[0]['masks'].shape[0]):
            pred = prediction[0]['masks'][m, 0].mul(255).byte().cpu().numpy()
            pred = np.stack((pred,)*3,axis=-1)
            r_color = rand_colors[m%100] 
            r = pred[:,:,0] 
            g = pred[:,:,1] 
            b = pred[:,:,2] 
            r[r!=0] = r_color[0]
            g[g!=0] = r_color[1]
            b[b!=0] = r_color[2]
            
            maskcolor = maskcolor| pred  
            imgmask = Image.fromarray(maskcolor)

        tmp = BytesIO()
        imgmask.save(tmp,format="png")
        tmp.seek(0)
        return base64.b64encode(tmp.read()).decode('ascii')

class SimpleHTTPRequestHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        """Serve a GET request."""
        print(self.path)
        if(self.path == '/code.js'):
            length = Path('code.js').stat().st_size
            codeJS = open('code.js','rb')
            self.send_response(200)
            self.send_header("Content-type", "text/javascript")
            self.send_header("Content-Length", str(length))
            self.end_headers()
            shutil.copyfileobj(codeJS,self.wfile)
            return

        f = self.send_head()
        if f:
            shutil.copyfileobj(f,self.wfile)
            f.close()
 
    def do_HEAD(self):
        """Serve a HEAD request."""
        f = self.send_head()
        if f:
            f.close()
 
    def do_POST(self):
        """Serve a POST request."""
        r, info,mask = self.deal_post_data()
        #print((r, info, "by: ", self.client_address))
        f = BytesIO()
        toWrite ={"base":info,"mask":mask} 
        f.write(json.dumps(toWrite).encode())
        length = f.tell()
        f.seek(0)
        self.send_response(200)
        self.send_header("Content-type", "application/json")
        self.send_header("Content-Length", str(length))
        self.end_headers()
        if f:
            shutil.copyfileobj(f,self.wfile)
            f.close()
        
    def deal_post_data(self):
        content_type = self.headers['content-type']
        if not content_type:
            return (False, "Content-Type header doesn't contain boundary")
        boundary = content_type.split("=")[1].encode()
        remainbytes = int(self.headers['content-length'])
        line = self.rfile.readline()
        remainbytes -= len(line)
        if not boundary in line:
            return (False, "Content NOT begin with boundary")
        line = self.rfile.readline()
        remainbytes -= len(line)
        fn = re.findall(r'Content-Disposition.*name="file"; filename="(.*)"', line.decode())
        if not fn:
            return (False, "Can't find out file name...")
        line = self.rfile.readline()
        remainbytes -= len(line)
        line = self.rfile.readline()
        remainbytes -= len(line)
        preline = self.rfile.readline()
        remainbytes -= len(preline)
        fileContent = b''
        while remainbytes > 0:
            line = self.rfile.readline()
            remainbytes -= len(line)
            if boundary in line:
                preline = preline[0:-1]
                if preline.endswith(b'\r'):
                    preline = preline[0:-1]
                fileContent += preline
                mask = ProduceMask(fileContent)
                fileContent = base64.b64encode(fileContent).decode('ascii')
                ft = fn[0].split('.')[1]
                return (True, f"data:image/{ft};base64,{fileContent}",f"data:image/png;base64,{mask}")
            else:
                #out.write(preline)
                fileContent += preline
                preline = line
        return (False, "Unexpect Ends of data.")
 
    def send_head(self):
        return self.list_directory()

    def list_directory(self):
        
        f = BytesIO()
        f.write(b'<!DOCTYPE html PUBLIC "-//W3C//DTD HTML 3.2 Final//EN">')
        f.write(("<html>\n<title>Mask your image</title>\n").encode())
        f.write(b"<head><script type=\"text/javascript\" src=\"code.js\"></script><style>img{width:200;height:200;} h2{font-famlily:tahoma;}</style></head>\n")
        f.write(("<body>\n<h2>Choose image for inference</h2>\n").encode())
        f.write(b"<hr>\n")
        f.write(b"<form ENCTYPE=\"multipart/form-data\" method=\"post\">")
        f.write(b"<input name=\"file\" type=\"file\"/>")
        f.write(b"<input onclick='InferAndShowResults(this)' type=\"button\" value=\"produce mask\"/></form>\n")
        f.write(b"</ul>\n<hr>\n</body>\n</html>\n")
        length = f.tell()
        f.seek(0)
        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.send_header("Content-Length", str(length))
        self.end_headers()
        return f
        
def test(HandlerClass = SimpleHTTPRequestHandler,
         ServerClass = http.server.HTTPServer,port = 8080):
    print(f"Serving forever on port {port}")
    ServerClass(('',port),HandlerClass).serve_forever()
    #http.server.test(HandlerClass, ServerClass)
 
if __name__ == '__main__':
    test(port = 8001)
