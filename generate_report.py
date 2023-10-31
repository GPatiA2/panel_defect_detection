from report_data_generator import ReportDataGenerator
from pannel_detector import PannelDetector
from pannel_chopper import PannelChopper
from models.model import PannelClassifier, NeuralClassifierLoader
from contourClassifier import ContourClassifier

from pylatex import Document, Section, Subsection, Tabular, Math, TikZ, Axis, \
    Plot, Figure, Matrix, Alignat, Command, LongTable, MultiColumn
from pylatex.utils import italic, NoEscape

COLORS = [
    "cyan",
    "yellow",
    "purple",
    "red",
    "lime",
    "orange"
]

detector   = PannelDetector('weights/model_final_thermal.pth')
chopper    = PannelChopper((70,100))
# classifier = NeuralClassifierLoader('results/try0/opt.json','results/try0/logs/lightning_logs/version_0/checkpoints/epoch=49-step=600.ckpt').load_classifier()
classifier = ContourClassifier({'rolling_iters':5, 'median_img':'median.png'})

gen = ReportDataGenerator(detector, classifier, chopper, 'dataset/generic_labels.json', 'latex_imgs', test=False)
gen.classes = ['healthy', 'defect']

report_data = gen.generate_report_data('datadron_real', show_crops=False)
images, total_defect_count = report_data
legend_path  = gen.save_legend_image()


geom_config = {"tmargin" : "2.5cm" ,
               "lmargin" : "5cm",
               "rmargin" : "5cm",
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
            im_fig.add_image(im[0], width=NoEscape(r'0.5\linewidth'))
            im_fig.add_caption(name)
        
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

    
        l += 1
        
        if l % 2 == 0:
            doc.append(NoEscape(r'\newpage'))



doc.generate_tex('report')







