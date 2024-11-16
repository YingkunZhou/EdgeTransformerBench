void evaluate(
#if defined(USE_NCNN)
    ncnn::Net &net,
    ncnn::Mat &input_tensor
#endif
#if defined(USE_MNN)
    std::shared_ptr<MNN::Interpreter> &net,
    MNN::Session *session,
    MNN::Tensor *input_tensor
#endif
#if defined(USE_TNN)
    tnn::TNN &net,
    std::shared_ptr<tnn::Instance> &instance,
    std::vector<float> &input
#endif
#if defined(USE_PDLITE)
    std::shared_ptr<paddle::lite_api::PaddlePredictor> &predictor,
    std::unique_ptr<paddle::lite_api::Tensor> &input_tensor
#endif
#if defined(USE_TFLITE)
    std::unique_ptr<Interpreter> &interpreter
#endif
#if defined(USE_ONNXRUNTIME)
    Ort::Session &session,
    std::vector<float> &input_tensor,
    std::vector<float> &output_tensor
#endif
#if defined(USE_TORCH)
#if defined(USE_TORCH_MOBILE)
    torch::jit::mobile::Module
#else
    torch::jit::script::Module
#endif
    &module,
    torch::Tensor &input
#endif
#if defined(USE_TVM)
    tvm::runtime::PackedFunc &set_input,
    tvm::runtime::PackedFunc &get_output,
    tvm::runtime::PackedFunc &run,
    tvm::runtime::NDArray &input_tensor,
    tvm::runtime::NDArray &output_tensor
#endif
#if defined(USE_OPENVINO)
    ov::InferRequest ireq,
    ov::Tensor input_tensor
#endif
)
{
    int class_index = 0;
    int num_predict = 0;
    int num_acc1 = 0;
    int num_acc5 = 0;
    std::cout << std::fixed << std::setprecision(4);

    int scale = 1;
    int offset = 0;
    if (args.data_path.find("20") != std::string::npos) {
        scale = 20;
    }
    else if (args.data_path.find("50") != std::string::npos) {
        scale = 50;
        offset = 15;
    }

    std::vector<std::filesystem::path> classes = traverse_class(args.data_path);
    auto start = high_resolution_clock::now();
    for (const std::string& class_path : classes) {
        for (const auto & image: std::filesystem::directory_iterator(class_path)) {
            num_predict++;
            int index = class_index*scale+offset;
#if defined(USE_NCNN)
            load_image(image.path(), (float *)input_tensor.data, args.model, args.input_size, args.batch_size);
            ncnn::Extractor ex = net.create_extractor();
            ex.input(args.input_name, input_tensor);
            ex.extract(args.output_name, args.output_tensor);
            num_acc5 += acck((float *)args.output_tensor.data, 5, index, num_acc1);
#endif
#if defined(USE_MNN)
            load_image(image.path(), input_tensor->host<float>(), args.model, args.input_size, args.batch_size);
            args.input->copyFromHostTensor(input_tensor);
            net->runSession(session);
            auto output_tensor = new MNN::Tensor(args.output, MNN::Tensor::CAFFE);
            args.output->copyToHostTensor(output_tensor);
            num_acc5 += acck(output_tensor->host<float>(), 5, index, num_acc1);
            delete output_tensor;
#endif
#if defined(USE_TNN)
            load_image(image.path(), input.data(), args.model, args.input_size, args.batch_size);
            auto input_tensor = std::make_shared<tnn::Mat>(tnn::DEVICE_NAIVE, tnn::NCHW_FLOAT, args.input_dims, input.data());
            args.status = instance->SetInputMat(input_tensor, tnn::MatConvertParam());
            args.status = instance->Forward();
            args.status = instance->GetOutputMat(args.output_tensor);
            num_acc5 += acck((float *)args.output_tensor->GetData(), 5, index, num_acc1);
#endif
#if defined(USE_PDLITE)
            load_image(image.path(), input_tensor->mutable_data<float>(), args.model, args.input_size, args.batch_size);
            predictor->Run();
            std::unique_ptr<const paddle::lite_api::Tensor> output_tensor(std::move(predictor->GetOutput(0)));
            num_acc5 += acck(output_tensor->data<float>(), 5, index, num_acc1);
#endif
#if defined(USE_TFLITE)
            float *input_tensor = interpreter->typed_input_tensor<float>(0);
            load_image(image.path(), input_tensor, args.model, args.input_size, args.batch_size);
            interpreter->Invoke();
            float *output_tensor = interpreter->typed_output_tensor<float>(0);
            num_acc5 += acck(output_tensor, 5, index, num_acc1);
#endif
#if defined(USE_ONNXRUNTIME)
            load_image(image.path(), input_tensor.data(), args.model, args.input_size, args.batch_size);
            session.Run(Ort::RunOptions{nullptr},
                        args.input_name.data(), args.input.data() , 1 /*Number of inputs*/,
                        args.output_name.data(),args.output.data(), 1 /*Number of outputs*/
                        );
            num_acc5 += acck(output_tensor.data(), 5, index, num_acc1);
#endif
#if defined(USE_TORCH)
            load_image(image.path(), input.data_ptr<float>(), args.model, args.input_size, args.batch_size);
            args.output = module.forward({input}).toTensor();
            num_acc5 += acck(args.output.data_ptr<float>(), 5, index, num_acc1);
#endif
#if defined(USE_TVM)
            load_image(image.path(), static_cast<float*>(input_tensor->data), args.model, args.input_size, args.batch_size);
            set_input("input", input_tensor);
            run();
            get_output(0, output_tensor);
            num_acc5 += acck(static_cast<float*>(output_tensor->data), 5, index, num_acc1);
#endif
#if defined(USE_OPENVINO)
            load_image(image.path(), static_cast<float*>(input_tensor.data()), args.model, args.input_size, args.batch_size);
            ireq.infer();
            ov::Tensor output_tensor = ireq.get_output_tensor();
            num_acc5 += acck(static_cast<float*>(output_tensor.data()), 5, index, num_acc1);
#endif
        }
        class_index++;
        std::cout << "Done [" << class_index << "/" << classes.size() << "]";
        std::cout << "\tacc1: " << num_acc1*1.f/num_predict;
        std::cout << "\tacc5: " << num_acc5*1.f/num_predict << std::endl;
    }
    auto stop = high_resolution_clock::now();
    std::cout << "elapse time: " << duration_cast<seconds>(stop - start).count() << std::endl;
}