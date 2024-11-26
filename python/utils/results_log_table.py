import sys
import numpy as np

meteorlake_win_idle = 2.84 #W
meteorlake_lin_idle = 1.32 #W
lunarlake_win_idle = 0.68 #W
lunarlake_lin_idle = 0.49 #W
m1mini_lin_idle = 4.96045 #W
m1mini_mac_idle = 1.81085 #W
power_bias = m1mini_mac_idle
android_offset = {
    'sd845': [1000000, 1000000],
    'sd888': [1000000, 1000000],
    'sd8gen2': [1000000, 1000],
    's20p': [1000000, -1000],
}


#define custom function
def g_mean(x):
    a = np.log(x)
    return np.exp(a.mean())

def avg_power(p_list):
    p_median = np.median(p_list)
    p_sum = 0
    p_cnt = 0
    for p in p_list:
        if abs(p - p_median) / p_median < 0.2:
            p_sum += p
            p_cnt += 1
    return p_sum / p_cnt

# (name, baseline)
model_list = [
    # SOTA Conv-ViT hybrid
    ('efficientformerv2_s0', 447.78),
    ('efficientformerv2_s1', 658.58),
    ('efficientformerv2_s2', 1052.78),
    ('', None),
    ('SwiftFormer_XS', 400.74),
    ('SwiftFormer_S', 538.97),
    ('SwiftFormer_L1', 751.41),
    ('', None),
    ('EMO_1M', 277.75),
    ('EMO_2M', 386.21),
    ('EMO_6M', 636.03),
    ('', None),
    ('edgenext_xx_small', 234.98),
    ('edgenext_x_small', 416.79),
    ('edgenext_small', 633.78),
    ('', None),
    ('mobilevitv2_050', 333.14),
    ('mobilevitv2_075', 622.92),
    ('mobilevitv2_100', 945.57),
    ('', None),
    ('mobilevit_xx_small', 347.99),
    ('mobilevit_x_small', 862.27),
    ('mobilevit_small', 1050.87),
    ('', None),
    ('LeViT_128S', 155.61),
    ('LeViT_128', 194.63),
    ('LeViT_192', 232.05),
    ('', None),
    ('', None),
    # conventional CNNs
    ('mobilenetv3_large_100', 144.67),
    ('tf_efficientnetv2_b0', 315.8),
    ('tf_efficientnetv2_b1', 478.43),
    ('tf_efficientnetv2_b2', 639.6),
    ('resnet50', 815.56),
]

model_name = ""
model_results = {}
csv_result = ""

def dump_csv():
    csv_result = "model,median,score,g-mean,power,a-mean,perf/W\n"
    scores = []
    powers = []
    tri_geomean = False
    tri_average = False

    # travel model list
    for ml in model_list:
        # unpack model name and baseline
        (name, baseline) = ml
        if name in model_results:
            latency = model_results[name][0]
            csv_result += f"{name},{latency},"
            # calculate score
            score = baseline/float(latency)
            scores.append(score)
            csv_result += "{:.2f},".format(score)
            if len(model_results[name]) > 1:
                power = float(model_results[name][1]) - power_bias
                powers.append(power)
                csv_result += ",{:.2f}".format(power)
                tri_average = True
            csv_result += '\n'
            tri_geomean = True
        elif name == '':
            if tri_geomean:
                # calculate model series (3 models) total score
                tri_geomean = False
                csv_result += ",,,{:.2f},".format(g_mean(scores[-3:]))
                if tri_average:
                    tri_average = False
                    csv_result += ",{:.2f}".format(np.average(powers[-3:]))
                    csv_result += ",{:.2f}".format(g_mean(scores[-3:])/np.average(powers[-3:]))
                csv_result += '\n'
            elif len(scores) == 21:
                # calculate 21 models total score
                csv_result += ",,,{:.2f},".format(g_mean(scores))
                if powers:
                    csv_result += ",{:.2f}".format(np.average(powers))
                csv_result += '\n'
            else:
                # missing certain models, don't caculate scores
                csv_result += ",,,,\n"
        else:
            csv_result += f"{name},,,,\n"
    return csv_result

if __name__ == "__main__":
    if len(sys.argv) != 2 and len(sys.argv) != 3:
        print("Usage: python results_log_table.py <log_file_path> #<output_csv_file>")
        sys.exit(1)
    log_file_path = sys.argv[1]
    output_csv_file =  sys.argv[2] if len(sys.argv) == 3 else log_file_path[:-3] + "csv"

    model_log = open(log_file_path).readlines()
    append_flag = False
    windows_logging = False
    x86linux_logging = False
    power_z_logging = False
    android_logging = False
    power_list = []
    for l in model_log:
        if ">>>>>>>>>>>" in l:
            if model_results != {}:
                if windows_logging and model_name in model_results and len(model_results[model_name]) == 1:
                    # for windows hwinfo, append the last model's power data
                    model_results[model_name].append(avg_power(power_list))
                # create a new benchmark iteration, dump the previous one
                csv_result += dump_csv()
                model_results = {}
            elif android_logging:
                # the first time
                power_bias = avg_power(power_list)
                print(power_bias)
        elif l.strip():
            l = l.strip()
            l_list = l.split()
            if len(l) > 3 and l[0:3] == 'AN-':
                android_logging = True
                ll = l.split(',')
                voltage = ll[-2]
                current = ll[-1]
                voltage = float(voltage)/android_offset[ll[0][3:]][0]
                current = float(current)/android_offset[ll[0][3:]][1]
                power_list.append(voltage * current)
            elif 'PkgWatt' in l:
                # intel linux turbostat format
                x86linux_logging = True
            elif 'power-z' in l:
                # power-z logging format
                power_z_logging = True
                power_bias = 0
                for end in model_log[-20:]:
                    power_bias += float(end.split(',')[-1])
                power_bias /= 20
                print(power_bias)
            elif l[0].isnumeric():
                # is the first one is number, which means it's the power log data
                # collect power_logs by scan each line
                if x86linux_logging:
                    power_list.append(float(l_list[0]))
                elif power_z_logging:
                    power_list.append(float(l.split(',')[-1]))
                elif ',' in l:
                    # windows hwinfo csv format
                    windows_logging = True
                    power_list.append(float(l.split(',')[2]))
                else:
                    # wall power meter format
                    power_list.append(float(l))
            elif l_list[0] == 'Load' or l_list[0] == 'Creating':
                # python api    or c++ api
                if windows_logging and model_name in model_results and len(model_results[model_name]) == 1:
                    # which means still need to append power data from windows hwinfo
                    model_results[model_name].append(avg_power(power_list))
                power_list = []
                # decide which models to append
                if l_list[-1] in [m[0] for m in model_list]:
                    append_flag = True
                    model_name = l_list[-1]
            elif android_logging and l_list[0] == "(index:":
                power_list = []
            elif "median" in l and append_flag:
                # to get the median latency (ms)
                append_flag = False
                # begin register name in model_results dictionary
                model_results[model_name] = [l_list[l_list.index("median")+2][:-2]]
                if power_z_logging:
                    model_results[model_name].append(avg_power(power_list[-20:]))
                    power_list = []
                elif power_list != []:
                    model_results[model_name].append(avg_power(power_list))
                    power_list = []

    if windows_logging and len(model_results[model_name]) == 1:
        # for windows hwinfo, append the last model's power data
        model_results[model_name].append(avg_power(power_list))
    # dump the last one
    csv_result += dump_csv()
    open(output_csv_file, "w").write(csv_result)