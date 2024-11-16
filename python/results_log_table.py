import sys

# (name, baseline)
model_list = [
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
    csv_result = "model,median\n"
    for ml in model_list:
        (name, baseline) = ml
        if name in model_results:
            latency = model_results[name]
            csv_result += f"{name},{latency},"
            csv_result += "{:.2f}\n".format(baseline/float(latency))
        elif name == '':
            csv_result += ",,\n"
        else:
            csv_result += f"{name},,\n"
    return csv_result

if __name__ == "__main__":
    if len(sys.argv) != 2 and len(sys.argv) != 3:
        print("Usage: python results_log_table.py <log_file_path> #<output_csv_file>")
        sys.exit(1)
    log_file_path = sys.argv[1]
    output_csv_file =  sys.argv[2] if len(sys.argv) == 3 else log_file_path[:-3] + "csv"

    model_log = open(log_file_path).readlines()
    append_flag = False
    for l in model_log:
        if ">>>>>>>>>>>" in l and model_results != {}:
            csv_result += dump_csv()
            model_results = {}
        elif l.strip():
            l = l.strip()
            tmp = l.split()[-1]
            if tmp != '' and tmp in [m[0] for m in model_list]:
                append_flag = True
                model_name = tmp
            if "median" in l and append_flag:
                data = l.split()
                model_results[model_name] = data[data.index("median")+2][:-2]
                append_flag = False
    csv_result += dump_csv()
    open(output_csv_file, "w").write(csv_result)