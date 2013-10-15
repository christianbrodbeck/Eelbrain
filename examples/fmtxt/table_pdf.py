from eelbrain import fmtxt
table = fmtxt.Table('lll')table.cell("Animal", r"\textbf")table.cell("Outside", r"\textbf")table.cell("Inside", r"\textbf")table.midrule()table.cell("Duck")table.cell("Feathers")table.cell("Duck Meat")table.cell("Dog")table.cell("Fur")table.cell("Hotdog Meat")
print table

# save the table as pdf with the following command:
# table.save_pdf('table.pdf')
