from report_data_generator import ReportDataGenerator
from pannel_detector import PannelDetector
from pannel_chopper import PannelChopper
from contourClassifier import ContourClassifier
from defectClassifier import DefectClassifier
from pylatex import Document, Section, Subsection, Tabular, Math, TikZ, Axis, \
    Plot, Figure, Matrix, Alignat, Command, LongTable, MultiColumn, SubFigure
from pylatex.utils import italic, NoEscape
import time
import argparse

def options():
    parser = argparse.ArgumentParser(description='Generate report from images')	
    parser.add_argument('--images', type=str, default='images', help='Path to images folder')
    parser.add_argument('--detector_weights', type=str, default='weights/model_final_thermal.pth', help='Path to detector weights')
    parser.add_argument('--defect_classifier_params', type=str, default='weights/defect_classifier_params.json', help='Path to defect classifier params')
    parser.add_argument('--crop_size', type=int, nargs= '+', help='Pannel Crop Size')
    parser.add_argument('--rolling_guidance_iters ', type=int, default=5, help='Rolling Guidance Iters')
    parser.add_argument('--median_img', type=str, default='median.png', help='Median Image')
    parser.add_argument('--report_images', type=str, default='latex_imgs', help='Path to report images folder')
    args = parser.parse_args()
    print(args)
    return args

args = options()

detector   = PannelDetector(args.detector_weights)
chopper    = PannelChopper(tuple(args.crop_size))
defect_detector = ContourClassifier({'rolling_iters':args.rolling_guidance_iters, 'median_img':args.median_img})
defect_classifier = DefectClassifier.load(args.defect_classifier_params)

start = time.time()
gen = ReportDataGenerator(detector, defect_detector, defect_classifier, chopper, args.report_images, test=False)

report_data = gen.generate_report_data(args.images, show_crops=False)
images, total_defect_count = report_data
legend_path  = gen.save_legend_image()


geom_config = {"tmargin" : "2.5cm" ,
               "lmargin" : "2.5cm",
               "rmargin" : "2.5cm",
               "bmargin" : "2.5cm"}

doc = Document(geometry_options=geom_config)

doc.preamble.append(Command('usepackage', 'float'))

doc.preamble.append(Command('title', 'Pannel defect report'))
doc.preamble.append(Command('author', 'CVAR Group - UPM'))
doc.preamble.append(Command('date', NoEscape(r'\today')))


doc.append(NoEscape(r'\maketitle'))
with doc.create(Figure(position='h!')) as logo:
    logo.add_image('latex_imgs/logo_CVAR.png', width=NoEscape(r'0.2\linewidth'))
    
with doc.create(Figure(position='h!')) as logo:
    logo.add_image('latex_imgs/logos_CAR.png', width=NoEscape(r'0.2\linewidth'))
doc.append(NoEscape(r'\newpage'))
doc.append(NoEscape(r'\tableofcontents'))
doc.append(NoEscape(r'\newpage'))

with doc.create(Section('Plant map')):
    with doc.create(Figure(position='h!')) as map_fig:
        map_fig.add_image('latex_imgs/plant_map.png', width=NoEscape(r'0.8\linewidth'))
        map_fig.add_caption('Plant map')

with doc.create(Section('Defect count')):
    with doc.create(LongTable('|l|l|')) as data_table:
        data_table.add_hline()
        data_table.add_row(['Defect', 'Count'])
        data_table.add_hline()
        data_table.end_table_header()
        data_table.add_hline()
        s = 0
        i = 0
        for key, val in total_defect_count.items():
            s += val
            data_table.add_row([key, val])
            i += 1

        data_table.add_hline()
        data_table.add_row(["Total", s])
        data_table.add_hline()

doc.append(NoEscape(r'\newpage'))

with doc.create(Section('Legend')):
    with doc.create(Figure(position='h!')) as legend_fig:
        legend_fig.add_image(legend_path, width=NoEscape(r'0.8\linewidth'))
        legend_fig.add_caption('Legend')

doc.append(NoEscape(r'\newpage'))

l = 0

with doc.create(Section('Defects')):
    for im in images:
        name = im[0].replace('latex_imgs/', '')

        with doc.create(Figure(position='h!')) as im_fig:
            with doc.create(SubFigure(position = 'r', width=NoEscape(r'0.6\linewidth'))) as image:
                image.add_image(im[0], width=NoEscape(r'0.55\linewidth'))
        
            with doc.create(SubFigure(position = 'l', width=NoEscape(r'0.4\linewidth'))) as table:
                with doc.create(LongTable('|l|l|', pos = 'c')) as im_table:
                    im_table.add_hline()
                    im_table.add_row((MultiColumn(2, align='|c|', data=name),))
                    im_table.add_hline()
                    im_table.add_row(['Defect', 'Count'])
                    im_table.add_hline()
                    im_table.end_table_header()
                    im_table.add_hline()
                    s = 0
                    for key, val in im[1].items():
                        s += val
                        im_table.add_row([key, val])
                    im_table.add_hline()
                    im_table.add_row(["Total", s])
                    im_table.add_hline()
            
            im_fig.add_caption(name)

    
        l += 1
        
        if l % 4 == 0:
            doc.append(NoEscape(r'\newpage'))



doc.generate_tex('report')
end = time.time()
print("Time Spent to Generate Report: ", end - start)







