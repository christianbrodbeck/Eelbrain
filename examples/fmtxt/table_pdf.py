from eelbrain import fmtxt
table = fmtxt.Table('lll')table.cell("Animal", "textbf")table.cell("Outside", "textbf")table.cell("Inside", "textbf")table.midrule()table.cell("Duck")table.cell("Feathers")table.cell("Duck Meat")table.cell("Dog")table.cell("Fur")table.cell("Hotdog Meat")
print table
table.save_pdf('table.pdf')
