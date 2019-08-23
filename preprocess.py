#!/usr/bin/env python
import sys, subprocess, argparse
import matplotlib.pyplot as plt

PASS_TRACES = "ethereum_traces/traces/pass/"
FAIL_BOOLEAN = "ethereum_traces/traces/fail_boolean/"
FAIL_RELATIONAL = "ethereum_traces/traces/fail_relational/"
FAIL_LOOP = "ethereum_traces/traces/fail_loop/"
FAIL_WRONG_NUMBERS = "ethereum_traces/traces/fail_wrong_numbers/"
FAIL_SWAPPED_ARGS = "ethereum_traces/traces/fail_swapped_args/"

REDUCED_PASS_TRACES = "ethereum_traces/reduced_traces/pass/"
REDUCED_FAIL_BOOLEAN = "ethereum_traces/reduced_traces/fail_boolean/"
REDUCED_FAIL_RELATIONAL = "ethereum_traces/reduced_traces/fail_relational/"
REDUCED_FAIL_LOOP = "ethereum_traces/reduced_traces/fail_loop/"
REDUCED_FAIL_WRONG_NUMBERS = "ethereum_traces/reduced_traces/fail_wrong_numbers/"
REDUCED_FAIL_SWAPPED_ARGS = "ethereum_traces/reduced_traces/fail_swapped_args/"

ENCODED_PASS_TRACES = "ethereum_traces/encoded_traces/pass/"
ENCODED_FAIL_BOOLEAN = "ethereum_traces/encoded_traces/fail_boolean/"
ENCODED_FAIL_RELATIONAL = "ethereum_traces/encoded_traces/fail_relational/"
ENCODED_FAIL_LOOP = "ethereum_traces/encoded_traces/fail_loop/"
ENCODED_FAIL_WRONG_NUMBERS = "ethereum_traces/encoded_traces/fail_wrong_numbers/"
ENCODED_FAIL_SWAPPED_ARGS = "ethereum_traces/encoded_traces/fail_swapped_args/"

ONE_HOT_THRESHOLD = 15000
AVE_POOL_KERNEL_SIZE = 10

def pool_trace(trace_path, target_path, num_of_tests):

	start_recording = False
	for i in range(1, num_of_tests + 1):
		postorder = []		
		trace = []
		start_recording = False
		switch = False
		f = open(trace_path + "DifficultyTest{}.log".format(i), 'r')
		for index, line in enumerate(f):
			try:
				caller = line.split('\n')[0].split(', ')[0].split(' ')[0]
				called = line.split('\n')[0].split(', ')[0].split(' ')[1]
			except IndexError:
				continue
			
			if "_ZN3dev3eth25calculateEthashDifficultyERKNS0_20ChainOperationParamsERKNS0_11BlockHeaderES6_" in caller:
				start_recording = True
			if start_recording == False or ("difficultyByzantium_invokerEv" in caller and "_ZNSt6localeD1Ev" in called):
				continue
			if "testDifficultyRKN5boost10filesystem4pathERN3dev3e" in caller and "calculateEthashDifficulty" in called:
				switch = True
				l_spl = line.split(' ')
				final_line = []
				for item in l_spl:
					if item != '0':
						final_line.append(item)
				postorder.append(' '.join(final_line))
				continue

			if switch == False:
				postorder.append(line)
			elif "main" in caller and "ret" in called:
				postorder.append(line)
			else:
				trace.append(line)

		for item in postorder:
			trace.append(item)

		with open(target_path + "DifficultyTest{}.log".format(i), 'w') as out:
			for line in trace[-20:]:
				out.write(line)

def preprocess_raw_data():

	trace_map = {'function_map': {}, 'function_size': 0}

	for file in range(1, 2255, 1):
		f = open(REDUCED_PASS_TRACES + "DifficultyTest" + str(file) + ".log", 'r')

		for line in f:
			if len(line.split(', ')) < 2: 
				continue

			line_spl = (line.replace(' \n', '\n').split('\n'))[0].split(', ')
			func_names = line_spl[0]
			caller = func_names.split(' ')[0]
			called = func_names.split(' ')[1]

			if caller not in trace_map['function_map']:
				trace_map['function_map'][caller] = {'index': 0, 'frequency': 1}
			else:
				trace_map['function_map'][caller]['frequency'] += 1

			if called not in trace_map['function_map']:
				trace_map['function_map'][called] = {'index': 0, 'frequency': 1}
			else:
				trace_map['function_map'][called]['frequency'] += 1
		f.close()

	for file in range(1, 2255, 1):
		f = open(REDUCED_FAIL_BOOLEAN + "DifficultyTest" + str(file) + ".log", 'r')

		for line in f:
			if len(line.split(', ')) < 2: 
				continue

			line_spl = (line.replace(' \n', '\n').split('\n'))[0].split(', ')
			func_names = line_spl[0]
			caller = func_names.split(' ')[0]
			called = func_names.split(' ')[1]

			if caller not in trace_map['function_map']:
				trace_map['function_map'][caller] = {'index': 0, 'frequency': 1}
			else:
				trace_map['function_map'][caller]['frequency'] += 1

			if called not in trace_map['function_map']:
				trace_map['function_map'][called] = {'index': 0, 'frequency': 1}
			else:
				trace_map['function_map'][called]['frequency'] += 1
		f.close()

	for file in range(1, 2255, 1):
		f = open(REDUCED_FAIL_BOOLEAN + "DifficultyTest" + str(file) + ".log", 'r')

		for line in f:
			if len(line.split(', ')) < 2:
				continue

			line_spl = (line.replace(' \n', '\n').split('\n'))[0].split(', ')
			func_names = line_spl[0]
			caller = func_names.split(' ')[0]
			called = func_names.split(' ')[1]

			if caller not in trace_map['function_map']:
				trace_map['function_map'][caller] = {'index': 0, 'frequency': 1}
			else:
				trace_map['function_map'][caller]['frequency'] += 1

			if called not in trace_map['function_map']:
				trace_map['function_map'][called] = {'index': 0, 'frequency': 1}
			else:
				trace_map['function_map'][called]['frequency'] += 1
		f.close()

	for file in range(1, 2255, 1):
		f = open(REDUCED_FAIL_RELATIONAL + "DifficultyTest" + str(file) + ".log", 'r')

		for line in f:
			if len(line.split(', ')) < 2: 
				continue

			line_spl = (line.replace(' \n', '\n').split('\n'))[0].split(', ')
			func_names = line_spl[0]
			caller = func_names.split(' ')[0]
			called = func_names.split(' ')[1]

			if caller not in trace_map['function_map']:
				trace_map['function_map'][caller] = {'index': 0, 'frequency': 1}
			else:
				trace_map['function_map'][caller]['frequency'] += 1

			if called not in trace_map['function_map']:
				trace_map['function_map'][called] = {'index': 0, 'frequency': 1}
			else:
				trace_map['function_map'][called]['frequency'] += 1
		f.close()

	for file in range(1, 2255, 1):
		f = open(REDUCED_FAIL_LOOP + "DifficultyTest" + str(file) + ".log", 'r')

		for line in f:
			if len(line.split(', ')) < 2: 
				continue

			line_spl = (line.replace(' \n', '\n').split('\n'))[0].split(', ')
			func_names = line_spl[0]
			caller = func_names.split(' ')[0]
			called = func_names.split(' ')[1]

			if caller not in trace_map['function_map']:
				trace_map['function_map'][caller] = {'index': 0, 'frequency': 1}
			else:
				trace_map['function_map'][caller]['frequency'] += 1

			if called not in trace_map['function_map']:
				trace_map['function_map'][called] = {'index': 0, 'frequency': 1}
			else:
				trace_map['function_map'][called]['frequency'] += 1
		f.close()

	for file in range(1, 2255, 1):
		f = open(REDUCED_FAIL_SWAPPED_ARGS + "DifficultyTest" + str(file) + ".log", 'r')

		for line in f:
			if len(line.split(', ')) < 2: 
				continue

			line_spl = (line.replace(' \n', '\n').split('\n'))[0].split(', ')
			func_names = line_spl[0]
			caller = func_names.split(' ')[0]
			called = func_names.split(' ')[1]

			if caller not in trace_map['function_map']:
				trace_map['function_map'][caller] = {'index': 0, 'frequency': 1}
			else:
				trace_map['function_map'][caller]['frequency'] += 1

			if called not in trace_map['function_map']:
				trace_map['function_map'][called] = {'index': 0, 'frequency': 1}
			else:
				trace_map['function_map'][called]['frequency'] += 1
		f.close()

	for file in range(1, 2255, 1):
		f = open(REDUCED_FAIL_WRONG_NUMBERS + "DifficultyTest" + str(file) + ".log", 'r')

		for line in f:
			if len(line.split(', ')) < 2: 
				continue

			line_spl = (line.replace(' \n', '\n').split('\n'))[0].split(', ')
			func_names = line_spl[0]
			caller = func_names.split(' ')[0]
			called = func_names.split(' ')[1]

			if caller not in trace_map['function_map']:
				trace_map['function_map'][caller] = {'index': 0, 'frequency': 1}
			else:
				trace_map['function_map'][caller]['frequency'] += 1

			if called not in trace_map['function_map']:
				trace_map['function_map'][called] = {'index': 0, 'frequency': 1}
			else:
				trace_map['function_map'][called]['frequency'] += 1
		f.close()

	keys_to_delete = []
	for func in trace_map['function_map']: 	
		if trace_map['function_map'][func]['frequency'] < ONE_HOT_THRESHOLD:
			keys_to_delete.append(func)

	for key in keys_to_delete:								
		del trace_map['function_map'][key]

	one_hot_index = 0
	for func in trace_map['function_map']:
		trace_map['function_map'][func]['index'] = one_hot_index
		one_hot_index += 1

	if ONE_HOT_THRESHOLD > 0:
		trace_map['function_map']['other'] = {'index': one_hot_index, 'frequency': 0} 

	trace_map['function_size'] = len(trace_map['function_map'])

	print("1-Hot encoding map")
	for func_map in trace_map['function_map']:
		print (trace_map['function_map'][func_map])

	return trace_map

def return_encoded_vector(size, index):
	encoded_list = ['0'] * size
	encoded_list[index] = '1'
	return encoded_list

def encode_string_to_bytes(input_string):

	encoded_string = []
	for char in input_string:
		encoded_string.append(str(ord(char)))

	assert(len(encoded_string) == len(input_string))
	return " ".join(encoded_string)


def convert_to_bin_vec(input_list):

	bin_list = []

	for item in input_list:
		try:
			item_list = item.split(' ')
			for num in item_list:
				int(num)
				str_num = str(bin(int(num)))

				for i in range(64 - len(str_num[2:])):
					bin_list.append('0')

				for char in str_num[2:]:
					bin_list.append(char)
		except ValueError:
			pass

	assert(len(bin_list) % 64 == 0)
	return bin_list

def encode_single_trace(trace_map, trace_path, output_path):

	f = open(trace_path, 'r')
	outfile = []

	for line in f:
		outline = []
		if len(line.split(', ')) < 2: 

			if len(line.split('\n')[0].split(' ')) > 1:
				outline.append(convert_to_bin_vec(line.replace(' \n', '\n').split('\n')))
				outfile.append(outline)
			continue
		line_spl = (line.replace(', ', ',').replace(' \n', '\n').split('\n'))[0].split(',')
		func_names = line_spl[0] #

		caller = func_names.split(' ')[0]
		called = func_names.split(' ')[1]

		if caller not in trace_map['function_map']:
			outline.append(return_encoded_vector(trace_map['function_size'], trace_map['function_map']['other']['index']))
		else:
			outline.append(return_encoded_vector(trace_map['function_size'], trace_map['function_map'][caller]['index']))

		if called not in trace_map['function_map']:
			outline[-1] += return_encoded_vector(trace_map['function_size'], trace_map['function_map']['other']['index'])
		else:
			outline[-1] += return_encoded_vector(trace_map['function_size'], trace_map['function_map'][called]['index'])

		ret_list = line_spl[1].replace('   ', ' ').replace('  ', ' ').split(' ') 

		try:
			arg_list = line_spl[2].replace('   ', ' ').replace('  ', ' ').split(' ')	
		except IndexError:
			arg_list = ['0']

		delete_me = []
		for item in ret_list:
			if (item == ''):
				delete_me.append(item)
		for item in delete_me:
			ret_list.remove(item)
		delete_me = []
		for item in arg_list:
			if item == '':
				delete_me.append(item)
		for item in delete_me:
			arg_list.remove(item)

		if len(ret_list) == 0:
			ret_list.append('0')
		if len(arg_list) == 0:
			arg_list.append('0')

		for index, item in enumerate(ret_list):
			try:
				float(item)
			except ValueError:
				ret_list[index] = encode_string_to_bytes(item)

		for index, item in enumerate(arg_list):
			try:
				float(item)
			except ValueError:
				arg_list[index] = encode_string_to_bytes(item)

		outline.append(convert_to_bin_vec(ret_list))
		outline.append(convert_to_bin_vec(arg_list))

		padded_ret = convert_to_bin_vec(ret_list)
		padded_arg = convert_to_bin_vec(arg_list)
		assert(len(padded_ret) % 64 == 0)
		assert(len(padded_arg) % 64 == 0)

		outfile.append(outline)

	out = open(output_path, 'w')
	for line_out in outfile:
		l_str = []
		for section in line_out:
			l_str.append(' '.join(section))
		out.write(','.join(l_str) + '\n')

	out.close()
	f.close()

	return


def encode_traces(trace_map, test_path, target_path, test_number):

	for file in range(1, test_number + 1):
		encode_single_trace(trace_map, test_path + "DifficultyTest" + str(file) + ".log", target_path + "DifficultyTest" + str(file) + ".csv")
	return

def main():

	pool_trace(PASS_TRACES, REDUCED_PASS_TRACES, 2254)

	pool_trace(FAIL_BOOLEAN, REDUCED_FAIL_BOOLEAN, 2254)
	pool_trace(FAIL_RELATIONAL, REDUCED_FAIL_RELATIONAL, 2254)
	pool_trace(FAIL_LOOP, REDUCED_FAIL_LOOP, 2254)
	pool_trace(FAIL_WRONG_NUMBERS, REDUCED_FAIL_WRONG_NUMBERS, 2254)
	pool_trace(FAIL_SWAPPED_ARGS, REDUCED_FAIL_SWAPPED_ARGS, 2254)

	trace_map = preprocess_raw_data()
	encode_traces(trace_map, REDUCED_PASS_TRACES, ENCODED_PASS_TRACES, 2254)

	encode_traces(trace_map, REDUCED_FAIL_BOOLEAN, ENCODED_FAIL_BOOLEAN, 2254)
	encode_traces(trace_map, REDUCED_FAIL_RELATIONAL, ENCODED_FAIL_RELATIONAL, 2254)
	encode_traces(trace_map, REDUCED_FAIL_LOOP, ENCODED_FAIL_LOOP, 2254)
	encode_traces(trace_map, REDUCED_FAIL_WRONG_NUMBERS, ENCODED_FAIL_WRONG_NUMBERS, 2254)
	encode_traces(trace_map, REDUCED_FAIL_SWAPPED_ARGS, ENCODED_FAIL_SWAPPED_ARGS, 2254)

	return

if __name__ == "__main__":

	main()
sys.exit(0)
