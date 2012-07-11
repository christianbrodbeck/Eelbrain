from eelbrain import fmtxt

table = fmtxt.Table('lll')table.cell()table.cell("example 1")table.cell("example 2")table.midrule()table.cell("string")table.cell('???')table.cell('another string')table.cell("Number")table.cell(4.5)table.cell(2./3, fmt='%.4g')print table
