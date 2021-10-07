from os import read

from pymongo import MongoClient
from urllib.parse import quote_plus
import fitz
from bson.json_util import loads


def db():
    uri = "mongodb://%s:%s@%s:%s" % (quote_plus("admin"), quote_plus("1q2w3e4r5t~!@#$%"), "127.0.0.1", "57017")
    client = MongoClient(uri)
    db = client["sigmai"]

    print(db.user.find_one())

def pdf():
    pdf_file = fitz.open("/home/dong/tmp/zuowen/JUYE_F_00007.pdf")

    for page_num in range(1, len(pdf_file), 2):
        pdf_doc = fitz.Document()
        pdf_doc.insert_pdf(pdf_file, from_page=page_num - 1, to_page=page_num)
        pdf_doc.save("/home/dong/tmp/tmp.pdf")
        break

uri = "mongodb://%s:%s@%s:%s" % (quote_plus("admin"), quote_plus("1q2w3e4r5t~!@#$%"), "127.0.0.1", "57017")
client = MongoClient(uri)
db = client["sigmai"]

pipeline = '''
    [
        {
            "$match": {
                "clazz": "5ea43fc7ebf5f3a540e44f7d",
                "status": {
                    "$in": ["在籍在读", "借读"]
                }
            }
        },
        {
            "$addFields": {
                "sortNo": {
                    "$toInt": {
                        "$cond": {
                            "if": {
                                "$eq": ["", { "$ifNull": ["$no", ""] }]
                            },
                            "then": "1000000",
                            "else": "$no"
                        }
                    }
                },
                "clazzId": {
                    "$toObjectId": "$clazz"
                }
            }
        },
    
        {
            "$lookup": {
                "from": "clazz",
                "foreignField": "_id",
                "localField": "clazzId",
                "as": "clazz"
            }
        },
        
        {
            "$unwind": "$clazz"
        },
        {
            "$sort": {
                "sortNo": 1
            }
        },
        {
            "$project": {
                "name": 1,
                "school": "$clazz.school",
                "schoolabbr": "$clazz.schoolabbr",
                "clazzId": "$clazz._id",
                "clazzName": "$clazz.name",
                "clazzNameabbr": "$clazz.nameabbr",
                "grade": "$clazz.grade",
                "startyear": "$clazz.startyear"
            }
        }
    ]
'''

# data = db.user.aggregate(loads(pipeline))
# for row in data:
#     print(row)

img = open("/home/dong/tmp/2021-10-03_14-44.png", "rb").read()
doc = fitz.open("/home/dong/tmp/zuowen/JUYE_F_00007.pdf")

print(doc.xref_object(6))
f = open("/home/dong/tmp/zuowen/img/0/JUYE_F_00007.pdf-1.jpg", "rb")

with open("/home/dong/tmp/xx", "wb") as wf:
    wf.write(doc.xref_stream(6))

#doc.update_stream(6, f.read(), 0)
rect = doc[0].bound()
doc.delete_page(0)
*_, width, height = rect
new_page = doc.new_page(0, width = width, height=height)

new_page.insert_image(rect, stream=f.read())

bytes = doc.tobytes()

open("/home/dong/tmp/tmp.pdf", "wb").write(bytes)
