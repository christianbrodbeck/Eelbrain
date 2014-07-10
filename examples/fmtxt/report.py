from eelbrain import *

ds = datasets.get_uv()
path = 'test_report'

# create the initial report
report = fmtxt.Report("Test Report", "Test Author")

# add a new section to the report and add some text
s1 = report.add_section("The Big Analysis")
s1.append("This is an introductory paragraph adding some text. It's important to have some text. What follows is a table showing the first ten rows of the data, so that it may be examined thoroughly.")

# Add a table to the section
t = ds.as_table(cases=10, midrule=True, caption="The first ten rows of the data table.")
s1.append(t)

# add a subsection with more text and another table
s11 = s1.add_section("Averages")
s11.append("This is a subordinate section which will show the average of the data.")

ds_ = ds.aggregate("A%B", drop=['rm'])
t = ds_.as_table(cases=0, midrule=True, caption="Averages of the data.")
s11.append(t)

# add a second subsection
s12 = s1.add_section("Figure")
s12.append("And now finally we will show a figure")
print s12

# add a figure
image = report.add_image_figure("boxplot.svg", "Boxplot of all the data")
p = plot.uv.boxplot('fltvar', 'A%B', 'rm', ds=ds)
p.figure.savefig(image, format='svg')
p.close()

# print string representation and save
print report.get_str()
report.save_html(path)
