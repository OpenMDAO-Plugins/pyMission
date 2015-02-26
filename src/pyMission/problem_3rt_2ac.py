import numpy

##########################
# misc parameters
##########################

misc_data = {}
misc_data['cost/fuel'] = 0.2431 # $/lb
misc_data['turnaround'] = 1.0 # hr
misc_data['weight/pax'] = 84 * 9.81 # N

##########################
# a/c parameters
##########################

ac_data = {}
ac_data['existing_ac'] = ['B738', 'B777']
ac_data['new_ac'] = ['CRM', 'BWB']

#
ac_data['capacity', 'B738'] = 122
ac_data['capacity', 'B777'] = 207
ac_data['capacity', 'CRM'] = 300
ac_data['capacity', 'BWB'] = 400

#
ac_data['number', 'B738'] = 6
ac_data['number', 'B777'] = 6
ac_data['number', 'CRM'] = 6
ac_data['number', 'BWB'] = 6

# maintenance hours / block hours
ac_data['MH', 'B738'] = 0.948
ac_data['MH', 'B777'] = 0.866
ac_data['MH', 'CRM'] = 0.866
ac_data['MH', 'BWB'] = 0.866

# lb
ac_data['fuel', 'B738'] = numpy.array([1e6, 1e6, 32537.5])
ac_data['fuel', 'B777'] = numpy.array([208401.46, 158127.02, 66423.87])

# hr
ac_data['block time', 'B738'] = numpy.array([25, 25, 5.98])
ac_data['block time', 'B777'] = numpy.array([15.03, 12.04, 5.74])

# $
ac_data['flight cost no fuel', 'B738'] = numpy.array([1e6, 1e6, 35666.2])
ac_data['flight cost no fuel', 'B777'] = numpy.array([245935.33, 188377.59, 84668.31])
ac_data['flight cost no fuel', 'CRM'] = numpy.array([132211.25, 106841.61, 47655.19])
ac_data['flight cost no fuel', 'BWB'] = numpy.array([132211.25, 106841.61, 47655.19])

# $
ac_data['ticket price', 'B738'] = numpy.array([0, 0, 371.74])
ac_data['ticket price', 'B777'] = numpy.array([1446.25, 1109.12, 500.29])
ac_data['ticket price', 'CRM'] = numpy.array([1446.25, 1109.12, 500.29])
ac_data['ticket price', 'BWB'] = numpy.array([1446.25, 1109.12, 500.29])

##########################
# route parameters
##########################

rt_data = {}
rt_data['number'] = 3

# nm
rt_data['range'] = numpy.array([6997.58, 5546.45, 2508.94])

#
rt_data['demand'] = numpy.array([1200, 550, 700])

#
rt_data['num_elem'] = 250 * numpy.ones(3, int)
rt_data['num_cp'] = 50 * numpy.ones(3, int)
