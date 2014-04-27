import xml.etree.ElementTree as ET
import os
import chardet
#from django.utils.encoding import smart_str

def extractTextFromElement(elementName, buf):
    tree = ET.fromstring(buf)
    for child in tree:
        if child.tag == elementName:
	       return child.text.strip()
                                    
rootdir ='/home/grk/Dropbox/scanned pdfs/Filtered/'

for subdir, dirs, files in os.walk(rootdir):
    for fil in files:  
        filename = subdir+'/'+fil
        if filename.endswith('.xml'):
        	buf = ""
	        buf += open(filename, 'rU').read()
                #extract the conference name
	        conference_name = extractTextFromElement('source_name', buf)
                #create the path
                if(conference_name):
                        path = "/home/grk/cs410_project/parsed_text/" + conference_name
	        if(conference_name):
                        print path
                        try:
			    os.makedirs(path)
			except OSError:
			    pass
                        #create the file name
                        name_file = os.path.basename(filename)
                        txt_file_name = name_file.replace(".xml",".txt")
                        new_path = path + "/"+ txt_file_name
                        content = extractTextFromElement('text',buf)
                        fptr = open(new_path , 'w+')
                        #content = unicode(content, "utf-8", errors="ignore")
                        #encoding = chardet.detect(content)                   
                        if content:
                            print >> fptr, content.encode('ascii', 'ignore')
                        #print >> fptr, smart_str(content)
                        #print content
                        fptr.close()                           
                        
                                                                

