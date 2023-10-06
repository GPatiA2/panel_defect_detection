from report_data_generator import ReportDataGenerator

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

gen = ReportDataGenerator('model_final_thermal.pth',
                          'classifier_test.ckpt', 'results/Try1/labels.json', (70,100), 'latex_imgs', test=True)

report_data = gen.generate_report_data('dataset_generation/datadron_real')
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







