from docxtpl import DocxTemplate, InlineImage
from docx.shared import Mm
import time, datetime
from log import logger

def rapport(self):
    doc = DocxTemplate("cctr_template.docx")

    im_load = InlineImage(doc, image_descriptor='load.png', width=Mm(150), height=Mm(100))
    im_res_x = InlineImage(doc, image_descriptor='res_x.png', width=Mm(150), height=Mm(100))
    im_res_y = InlineImage(doc, image_descriptor='res_y.png', width=Mm(150), height=Mm(100))
    im_res_sum = InlineImage(doc, image_descriptor='res_sum.png', width=Mm(150), height=Mm(100))
    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime("%d/%m/%y - %H:%M")
    short_st = datetime.datetime.fromtimestamp(ts).strftime("%d_%m_%M%H")

    res = self.res

    context = {'date': st,
               'bois': 'C24',
               'var': 30,
               'Image':
                   {'load': im_load,
                    'res_x': im_res_x,
                    'res_y': im_res_y,
                    'res_sum': im_res_sum
                    },
               'res': res
               }

    doc.render(context)
    doc.save("Rapport_" + short_st + ".docx")
    return print("Rapport genéré avec succès")