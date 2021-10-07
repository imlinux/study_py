from pymongo import MongoClient
from urllib.parse import quote_plus
import gridfs

def open_db():
    uri = "mongodb://%s:%s@%s:%s" % (quote_plus("admin"), quote_plus("1q2w3e4r5t~!@#$%"), "127.0.0.1", "57017")
    client = MongoClient(uri)
    db = client["sigmai"]

    return db, gridfs.GridFS(db)



db, fs = open_db()

print(fs.find_one({"filename": "wxd4bfc3f669350acf.o6zAJs4_zLr3N65iMlNj56YeKbo4.nPG0Dtsy9bNF1c2bf771d832c59e44197f0595bf4ba2.jpeg1"}))

# data = open("/home/dong/tmp/tmp.pdf", "rb").read()
#
# id = fs.put(data, filename="测试.pdf")
#
# print(id)

