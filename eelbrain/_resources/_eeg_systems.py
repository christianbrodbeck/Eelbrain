# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>

connectivity_data = {'BrainCap32Ch': """
FP1	VEOGt	F3	F7
VEOGt	F4	F8
F7	F3	FC5
F8	F4	FC6
FC5	F3	C3
F3	FZ	FC1
FZ	F4	FC1	FCZ	FC2
F4	FC2	FC6
FC6	C4
FC1	C3	CZ	FCZ
FCZ	CZ	FC2
FC2	CZ	C4
CP5	C3	P3	P7
C3	CP1	CZ
CZ	CP1	CPZ	CP2	C4
C4	CP2	CP6
CP6	P4	P8
CP1	P3	PZ	CPZ
CPZ	PZ	CP2
CP2	PZ	P4
P7	A1	O1	P3
P3	O1	POZ	PZ
PZ	POZ	P4
P4	POZ	O2	P8
P8	A2	O2
O1	POZ	O2
POZ	O2
"""}


def predefined_connectivity(name):
    connectivity = []
    for line in connectivity_data[name].split('\n'):
        items = line.split('\t')
        src = items[0]
        for dst in items[1:]:
            connectivity.append((src, dst))
    return connectivity
