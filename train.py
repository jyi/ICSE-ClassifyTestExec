#!/usr/bin/env python
import sys, argparse
import torch
import torch.nn as nn
from random import randint
from random import shuffle
import torch.optim as optim

CODEBASE_NAME = "DifficultyTest"
ONE_HOT_FUNCS_SIZE = 0

PASS_TRACES = "ethereum_traces/encoded_traces/pass/"
FAIL_BOOLEAN = "ethereum_traces/encoded_traces/fail_boolean/"
FAIL_RELATIONAL = "ethereum_traces/encoded_traces/fail_relational/"
FAIL_LOOP = "ethereum_traces/encoded_traces/fail_loop/"
FAIL_SWAPPED_ARGS = "ethereum_traces/encoded_traces/fail_swapped_args/"
FAIL_WRONG_NUMBERS = "ethereum_traces/encoded_traces/fail_wrong_numbers/"

def generate_dataset():

	dataset = []
	unseen = []

	for i in range (1, 2255):
		datapoint, global_values = process_encoded_trace(PASS_TRACES + CODEBASE_NAME + str(i) + ".csv")
		dataset.append({'function_calls': datapoint, 'globals': global_values, 'label': 1, 'index': i})

	for i in range (1, 452):
		datapoint, global_values = process_encoded_trace(FAIL_BOOLEAN + CODEBASE_NAME + str(i) + ".csv")
		dataset.append({'function_calls': datapoint, 'globals': global_values, 'label': 0, 'index': i})

	for i in range (1, 451):
		datapoint, global_values = process_encoded_trace(FAIL_RELATIONAL + CODEBASE_NAME + str(i) + ".csv")
		dataset.append({'function_calls': datapoint, 'globals': global_values, 'label': 0, 'index': i})

	for i in range (1, 451):
		datapoint, global_values = process_encoded_trace(FAIL_LOOP + CODEBASE_NAME + str(i) + ".csv")
		dataset.append({'function_calls': datapoint, 'globals': global_values, 'label': 0, 'index': i})


	for i in range (1, 451):
		datapoint, global_values = process_encoded_trace(FAIL_SWAPPED_ARGS + CODEBASE_NAME + str(i) + ".csv")
		dataset.append({'function_calls': datapoint, 'globals': global_values, 'label': 0, 'index': i})


	for i in range (1, 451):
		datapoint, global_values = process_encoded_trace(FAIL_WRONG_NUMBERS + CODEBASE_NAME + str(i) + ".csv")
		dataset.append({'function_calls': datapoint, 'globals': global_values, 'label': 0, 'index': i})


	for i in range (452, 2255):
		datapoint, global_values = process_encoded_trace(FAIL_BOOLEAN + CODEBASE_NAME + str(i) + ".csv")
		unseen.append({'function_calls': datapoint, 'globals': global_values, 'label': 0, 'index': i})

	for i in range (451, 2255):
		datapoint, global_values = process_encoded_trace(FAIL_RELATIONAL + CODEBASE_NAME + str(i) + ".csv")
		unseen.append({'function_calls': datapoint, 'globals': global_values, 'label': 0, 'index': i})

	for i in range (451, 2255):
		datapoint, global_values = process_encoded_trace(FAIL_LOOP + CODEBASE_NAME + str(i) + ".csv")
		unseen.append({'function_calls': datapoint, 'globals': global_values, 'label': 0, 'index': i})


	for i in range (451, 2255):
		datapoint, global_values = process_encoded_trace(FAIL_SWAPPED_ARGS + CODEBASE_NAME + str(i) + ".csv")
		unseen.append({'function_calls': datapoint, 'globals': global_values, 'label': 0, 'index': i})


	for i in range (451, 2255):
		datapoint, global_values = process_encoded_trace(FAIL_WRONG_NUMBERS + CODEBASE_NAME + str(i) + ".csv")
		unseen.append({'function_calls': datapoint, 'globals': global_values, 'label': 0, 'index': i})

	global ONE_HOT_FUNCS_SIZE
	ONE_HOT_FUNCS_SIZE = len(dataset[0]['function_calls'][0]['funcs'])

	return dataset, unseen

def process_encoded_trace(path):

	datapoint = []
	global_values = []

	f = open(path, 'r')
	for line in f:

		line_split = line.replace('\n', '').split(',')
		if len(line_split) > 1:
			datapoint.append({'funcs': line_split[0].split(' '), 'ret': line_split[1].split(' '), 'arg': line_split[2].split(' ')})
		else:
			global_values.append(line_split[0].split(' '))

	global ONE_HOT_FUNCS_SIZE
	ONE_HOT_FUNCS_SIZE = len(datapoint[0]['funcs'])

	return datapoint, global_values

def test_tensor(tensor):
	assert torch.isnan(tensor).any() == 0
	return

def custom_pooling(input, line_size):

	channels = input.size(0)
	summed_tensor = torch.zeros([1, line_size], dtype = torch.float)

	for _x in input:
		summed_tensor += _x

	summed_tensor /= channels
	print(summed_tensor.size())

	return

def train_and_deploy():

	datapoints, unseen = generate_dataset()

	encoding_lstm_size = 64
	encoding_ret_lstm = 64
	line_lstm_size = 128
	line_size = encoding_lstm_size*2 + ONE_HOT_FUNCS_SIZE
	global_lstm_size = line_size
	fc1 = 128
	fc2 = 64
	fc3 = 32

	ret_lstm = nn.LSTM(input_size = 64, hidden_size = encoding_ret_lstm, num_layers = 1)
	arg_lstm = nn.LSTM(input_size = 64, hidden_size = encoding_lstm_size, num_layers = 1)
	global_lstm = nn.LSTM(input_size = 64, hidden_size = global_lstm_size, num_layers = 1)
	final_lstm = nn.LSTM(input_size = line_size, hidden_size = line_lstm_size, num_layers = 1)
	feed_input = nn.Linear(line_lstm_size, fc1)
	feed_hidden = nn.Linear(fc1, fc2)
	feed_output = nn.Linear(fc2, fc3)
	feed_output2 = nn.Linear(fc3, 1)
	log_softmax = nn.LogSoftmax()

	loss_fn = nn.CrossEntropyLoss()
	loss_bce = nn.BCELoss()
	loss_sigm_bce = nn.BCEWithLogitsLoss()
	loss_function = nn.NLLLoss()

	optimizer_Adam = torch.optim.Adam(list(ret_lstm.parameters())
										+ list(arg_lstm.parameters())
										+ list(global_lstm.parameters())
										+ list(final_lstm.parameters())
										+ list(feed_input.parameters())
										+ list(feed_hidden.parameters())
										+ list(feed_output.parameters())
										+ list(feed_output2.parameters())
										, lr = 0.000008)
	optimizer_SGD = torch.optim.SGD(list(ret_lstm.parameters())
										+ list(arg_lstm.parameters())
										+ list(global_lstm.parameters())
										+ list(final_lstm.parameters())
										+ list(feed_input.parameters())
										+ list(feed_hidden.parameters())
										+ list(feed_output.parameters())
										+ list(feed_output2.parameters())
										, lr = 0.000008)

	with torch.no_grad():
		func_str_list = datapoints[0]['function_calls'][0]['funcs']
		func_list = list(map(float, func_str_list))
		ret_str_list = datapoints[0]['function_calls'][0]['ret']
		ret_list = list(map(float, ret_str_list))
		arg_str_list = datapoints[0]['function_calls'][0]['arg']
		arg_list = list(map(float, arg_str_list))

		func_tensor = torch.FloatTensor(func_list)
		func_tensor = func_tensor.unsqueeze(0).unsqueeze(0)

		ret_tensor = torch.randn(int(len(ret_list) / 64), 64)
		for index, item in enumerate(ret_list):
			ret_tensor[int(index / 64)][index % 64] = item
		ret_tensor = ret_tensor.unsqueeze(1)

		arg_tensor = torch.randn(int(len(ret_list) / 64), 64)
		for index, item in enumerate(ret_list):
			arg_tensor[int(index / 64)][index % 64] = item
		arg_tensor = arg_tensor.unsqueeze(1)

		h0_ret = torch.randn(1*1, 1, encoding_lstm_size)
		c0_ret = torch.randn(1*1, 1, encoding_lstm_size)
		h0_arg = torch.randn(1*1, 1, encoding_lstm_size)
		c0_arg = torch.randn(1*1, 1, encoding_lstm_size)
		h0_final = torch.randn(1*1, 1, line_lstm_size)
		c0_final = torch.randn(1*1, 1, line_lstm_size)

		print(ret_tensor.size())
		out_ret, hn_ret = ret_lstm(ret_tensor, (h0_ret, c0_ret))
		out_arg, hn_arg = arg_lstm(arg_tensor, (h0_arg, c0_arg))

		full_line = torch.cat((torch.cat((func_tensor, out_ret[-1].unsqueeze(1)), 2), out_arg[-1].unsqueeze(1)), 2)

		out_line, hn = final_lstm(full_line, (h0_final, c0_final))
		out = feed_output2(feed_output(feed_hidden(feed_input(out_line))))

	epochs = 40

	shuffled_dataset = datapoints
	for i in range(100):
		shuffle(shuffled_dataset)

	training_length = 1400
	training_set = shuffled_dataset[:training_length]
	evaluation_set  = shuffled_dataset[training_length:]

	evaluation_set += unseen

	visualize_pass = []
	visualize_fail = []
	for x in range(epochs):
		print("Epoch: " + str(x))
		for i in range(100):
			shuffle(training_set)
		for tr_index, input_data in enumerate(training_set):

			input_list = []
			for func_line in input_data['function_calls']:

				func_str_list = func_line['funcs']
				func_list = list(map(float, func_str_list))
				func_tensor = torch.FloatTensor(func_list)
				func_tensor = func_tensor.unsqueeze(0).unsqueeze(0) 
				ret_str_list = func_line['ret']
				ret_list = list(map(float, ret_str_list))
				arg_str_list = func_line['arg']
				arg_list = list(map(float, arg_str_list))		
				ret_tensor = torch.randn(int(len(ret_list) / 64), 64)
				for index, item in enumerate(ret_list):
					ret_tensor[int(index / 64)][index % 64] = item
				ret_tensor = ret_tensor.unsqueeze(1)
				arg_tensor = torch.randn(int(len(arg_list) / 64), 64)
				for index, item in enumerate(arg_list):
					arg_tensor[int(index / 64)][index % 64] = item
				arg_tensor = arg_tensor.unsqueeze(1)

				test_tensor(func_tensor)
				test_tensor(ret_tensor)
				test_tensor(arg_tensor)

				h0_ret = torch.randn(1*1, 1, encoding_lstm_size)
				c0_ret = torch.randn(1*1, 1, encoding_lstm_size)

				h0_arg = torch.randn(1*1, 1, encoding_lstm_size)
				c0_arg = torch.randn(1*1, 1, encoding_lstm_size)

				out_ret, hn_ret = ret_lstm(ret_tensor, (h0_ret, c0_ret))
				test_tensor(out_ret)
				out_arg, hn_arg = arg_lstm(arg_tensor, (h0_arg, c0_arg)) 
				test_tensor(out_arg)
				encoded_line = torch.cat((torch.cat((func_tensor, out_ret[-1].unsqueeze(1)), 2), out_arg[-1].unsqueeze(1)), 2).squeeze(0)
				test_tensor(encoded_line)

				input_list.append(encoded_line)

			for glob in input_data['globals']:

				glob_list = list(map(float, glob))
				glob_tensor = torch.randn(int(len(glob_list) / 64), 64)

				for index, item in enumerate(glob_list):
					glob_tensor[int(index / 64)][index % 64] = item
				glob_tensor = glob_tensor.unsqueeze(1)				
				test_tensor(glob_tensor)

				h0_glob = torch.randn(1*1, 1, global_lstm_size)
				c0_glob = torch.randn(1*1, 1, global_lstm_size)

				out_glob, hn_glob = global_lstm(glob_tensor, (h0_glob, c0_glob))
				test_tensor(out_glob)
				input_list.append(out_glob[-1])

			input_tensor = torch.randn(len(input_list), 1, line_size)
			for index, line in enumerate(input_list):
				input_tensor[index] = line

			h0_final = torch.randn(1*1, 1, line_lstm_size)
			c0_final = torch.randn(1*1, 1, line_lstm_size)
			out, hn = final_lstm(input_tensor, (h0_final, c0_final))
			test_tensor(out)
			scores = feed_output2(feed_output(feed_hidden(feed_input(out[-1]))))
			test_tensor(scores)

			if input_data['label'] == 1:
				target = torch.tensor([[1.]])
				loss = loss_sigm_bce(scores, target)
				loss.backward()
				optimizer_Adam.step()
				if (tr_index % 10) == 0:
					print("Pass Loss: {:6.4f}, {}".format(loss.item(), tr_index))
			else:
				target = torch.tensor([[0.]])
				loss = loss_sigm_bce(scores, target)
				loss.backward()
				optimizer_Adam.step()
				if (tr_index % 10) == 0:
					print("Fail Loss: {:6.4f}, {}".format(loss.item(), tr_index))
					
		with torch.no_grad():
			pos_matches, neg_matches, pos_total, neg_total = 0, 0, 0, 0
			for data_point in training_set:

				input_list = []
				for func_line in data_point['function_calls']:
					func_str_list = func_line['funcs']
					func_list = list(map(float, func_str_list))
					func_tensor = torch.FloatTensor(func_list)
					func_tensor = func_tensor.unsqueeze(0).unsqueeze(0)
					ret_str_list = func_line['ret']
					ret_list = list(map(float, ret_str_list))
					arg_str_list = func_line['arg']
					arg_list = list(map(float, arg_str_list))
					ret_tensor = torch.randn(int(len(ret_list) / 64), 64)
					for index, item in enumerate(ret_list):
						ret_tensor[int(index / 64)][index % 64] = item
					ret_tensor = ret_tensor.unsqueeze(1)

					arg_tensor = torch.randn(int(len(arg_list) / 64), 64)
					for index, item in enumerate(arg_list):
						arg_tensor[int(index / 64)][index % 64] = item
					arg_tensor = arg_tensor.unsqueeze(1)

					test_tensor(func_tensor)
					test_tensor(ret_tensor)
					test_tensor(arg_tensor)
	
					h0_ret = torch.randn(1*1, 1, encoding_lstm_size)
					c0_ret = torch.randn(1*1, 1, encoding_lstm_size)

					h0_arg = torch.randn(1*1, 1, encoding_lstm_size)
					c0_arg = torch.randn(1*1, 1, encoding_lstm_size)
					out_ret, hn_ret = ret_lstm(ret_tensor, (h0_ret, c0_ret))
					test_tensor(out_ret)
					out_arg, hn_arg = arg_lstm(arg_tensor, (h0_arg, c0_arg)) 
					test_tensor(out_arg)
					encoded_line = torch.cat((torch.cat((func_tensor, out_ret[-1].unsqueeze(1)), 2), out_arg[-1].unsqueeze(1)), 2).squeeze(0)
					test_tensor(encoded_line)
					input_list.append(encoded_line)

				for glob in input_data['globals']:

					glob_list = list(map(float, glob))

					glob_tensor = torch.randn(int(len(glob_list) / 64), 64)
					for index, item in enumerate(glob_list):
						glob_tensor[int(index / 64)][index % 64] = item
					glob_tensor = glob_tensor.unsqueeze(1)				
					test_tensor(glob_tensor)

					h0_glob = torch.randn(1*1, 1, global_lstm_size)
					c0_glob = torch.randn(1*1, 1, global_lstm_size)

					out_glob, hn_glob = global_lstm(glob_tensor, (h0_glob, c0_glob))
					test_tensor(out_glob)
					input_list.append(out_glob[-1])

				input_tensor = torch.randn(len(input_list), 1, line_size)
				for index, line in enumerate(input_list):
					input_tensor[index] = line

				out, hn = final_lstm(input_tensor, (h0_final, c0_final))
				test_tensor(out)

				scores = feed_output2(feed_output(feed_hidden(feed_input(out[-1])))) 
				test_tensor(scores)
				normalized_score = torch.sigmoid(scores)
				test_tensor(normalized_score)

				if (data_point['label'] == 1):
					pos_total += 1
					if normalized_score.item() >= 0.5:
						pos_matches += 1
				else:
					neg_total += 1
					if normalized_score.item() <= 0.5:
						neg_matches += 1

			try:
				accuracy = (neg_matches + pos_matches) / (neg_total + pos_total)
			except ZeroDivisionError:
				accuracy = -1
				print("Zero division on accuracy: {}, {}, {}, {}".format(neg_matches, pos_matches, neg_total, pos_total))
			try:
				precision = pos_matches / pos_total
			except ZeroDivisionError:
				precision = -1
				print("Zero division on precision: {0}. {1}".format(pos_matches, pos_total))
			
			try:
				neg_precision = neg_matches / neg_total
			except ZeroDivisionError:
				neg_precision = -1
				print("Zero division on neg precision: {0} {1}".format(neg_matches, neg_total))

			try:
				recall = pos_matches / ((neg_total - neg_matches) + pos_matches)
			except ZeroDivisionError:
				recall = -1
				print("Zero division on recall: {0}, {1}, {2}".format(pos_matches, neg_total, neg_matches))

			try:
				false_alarm_rate = (pos_total - pos_matches) / ((pos_total - pos_matches) + neg_matches)
			except ZeroDivisionError:
				false_alarm_rate = -1
				print("Zero Division on false alarm rate {0}, {1}, {2}".format(pos_total, pos_matches, neg_matches))
			try:
				miss_rate = (neg_total - neg_matches) / ((neg_total - neg_matches) + pos_matches)
			except ZeroDivisionError:
				miss_rate = -1
				print("Zero division on miss rate: {0}, {1}. {2}".format(neg_total, neg_matches, pos_matches))

			print("Epoch {} Training set:----------------------------------".format(x))
			print("Accuracy: {:4.2f}%, Pass precision: {:4.2f}%, Fail precision: {:4.2f}%, Recall: {:4.2f}%, False alarm rate: {:4.2f}%, Miss rate: {:4.2f}%".format(accuracy * 100, precision * 100, neg_precision * 100, recall * 100, false_alarm_rate * 100, miss_rate * 100))

		with torch.no_grad():
			pos_matches, neg_matches, pos_total, neg_total = 0, 0, 0, 0
			for data_point in evaluation_set:

				input_list = []
				for func_line in data_point['function_calls']:

					func_str_list = func_line['funcs']
					func_list = list(map(float, func_str_list))
					func_tensor = torch.FloatTensor(func_list)
					func_tensor = func_tensor.unsqueeze(0).unsqueeze(0)
					ret_str_list = func_line['ret']
					ret_list = list(map(float, ret_str_list))
					arg_str_list = func_line['arg']
					arg_list = list(map(float, arg_str_list))
					ret_tensor = torch.randn(int(len(ret_list) / 64), 64)
					for index, item in enumerate(ret_list):
						ret_tensor[int(index / 64)][index % 64] = item
					ret_tensor = ret_tensor.unsqueeze(1)
					arg_tensor = torch.randn(int(len(arg_list) / 64), 64)
					for index, item in enumerate(arg_list):
						arg_tensor[int(index / 64)][index % 64] = item
					arg_tensor = arg_tensor.unsqueeze(1)
					test_tensor(func_tensor)
					test_tensor(ret_tensor)
					test_tensor(arg_tensor)

					h0_ret = torch.randn(1*1, 1, encoding_lstm_size)
					c0_ret = torch.randn(1*1, 1, encoding_lstm_size)
					h0_arg = torch.randn(1*1, 1, encoding_lstm_size)
					c0_arg = torch.randn(1*1, 1, encoding_lstm_size)

					out_ret, hn_ret = ret_lstm(ret_tensor, (h0_ret, c0_ret))
					test_tensor(out_ret)
					out_arg, hn_arg = arg_lstm(arg_tensor, (h0_arg, c0_arg))  
					test_tensor(out_arg)
					encoded_line = torch.cat((torch.cat((func_tensor, out_ret[-1].unsqueeze(1)), 2), out_arg[-1].unsqueeze(1)), 2).squeeze(0)
					test_tensor(encoded_line)
					input_list.append(encoded_line)

				for glob in input_data['globals']:

					glob_list = list(map(float, glob))

					glob_tensor = torch.randn(int(len(glob_list) / 64), 64)
					for index, item in enumerate(glob_list):
						glob_tensor[int(index / 64)][index % 64] = item
					glob_tensor = glob_tensor.unsqueeze(1)				
					test_tensor(glob_tensor)

					h0_glob = torch.randn(1*1, 1, global_lstm_size)
					c0_glob = torch.randn(1*1, 1, global_lstm_size)

					out_glob, hn_glob = global_lstm(glob_tensor, (h0_glob, c0_glob))
					test_tensor(out_glob)
					input_list.append(out_glob[-1])

				input_tensor = torch.randn(len(input_list), 1, line_size)
				for index, line in enumerate(input_list):
					input_tensor[index] = line

				out, hn = final_lstm(input_tensor, (h0_final, c0_final))
				test_tensor(out)
				scores = feed_output2(feed_output(feed_hidden(feed_input(out[-1]))))
				test_tensor(scores)
				normalized_score = torch.sigmoid(scores)
				test_tensor(normalized_score)

				if (data_point['label'] == 1):
					pos_total += 1
					if normalized_score.item() >= 0.5:
						pos_matches += 1
				else:
					neg_total += 1
					if normalized_score.item() <= 0.5:
						neg_matches += 1

			try:
				accuracy = (neg_matches + pos_matches) / (neg_total + pos_total)
			except ZeroDivisionError:
				accuracy = -1
				print("Zero division on accuracy: {}, {}, {}, {}".format(neg_matches, pos_matches, neg_total, pos_total))

			try:
				precision = pos_matches / pos_total
			except ZeroDivisionError:
				precision = -1
				print("Zero division on precision: {0}. {1}".format(pos_matches, pos_total))

			try:
				neg_precision = neg_matches / neg_total
			except ZeroDivisionError:
				neg_precision = -1
				print("Zero division on neg precision: {0} {1}".format(neg_matches, neg_total))

			try:
				recall = pos_matches / ((neg_total - neg_matches) + pos_matches)
			except ZeroDivisionError:
				recall = -1
				print("Zero division on recall: {0}, {1}, {2}".format(pos_matches, neg_total, neg_matches))

			try:
				false_alarm_rate = (pos_total - pos_matches) / ((pos_total - pos_matches) + neg_matches)
			except ZeroDivisionError:
				false_alarm_rate = -1
				print("Zero Division on false alarm rate {0}, {1}, {2}".format(pos_total, pos_matches, neg_matches))

			try:
				miss_rate = (neg_total - neg_matches) / ((neg_total - neg_matches) + pos_matches)
			except ZeroDivisionError:
				miss_rate = -1
				print("Zero division on miss rate: {0}, {1}. {2}".format(neg_total, neg_matches, pos_matches))

			print("Epoch {} Validation set:----------------------------------".format(x))
			print("Accuracy: {:4.2f}%, Pass precision: {:4.2f}%, Fail precision: {:4.2f}%, Recall: {:4.2f}%, False alarm rate: {:4.2f}%, Miss rate: {:4.2f}%".format(accuracy * 100, precision * 100, neg_precision * 100, recall * 100, false_alarm_rate * 100, miss_rate * 100))

	return

def main():
	train_and_deploy()
	return

if __name__ == "__main__":

	main()
