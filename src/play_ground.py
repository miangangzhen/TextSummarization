from tensorflow.python.tools import inspect_checkpoint as chkp

if __name__ == "__main__":
    model = "E:\\pointer-generator\\chinese_summary\\train\\model.ckpt-22698"
    model = "E:\\TextSummarization\\model\\model.ckpt-0"
    chkp.print_tensors_in_checkpoint_file(file_name=model,
                                      tensor_name=None,  # 如果为None,则默认为ckpt里的所有变量
                                      all_tensors=False,  # bool 是否打印所有的tensor，这里打印出的是tensor的值，一般不推荐这里设置为False
                                      all_tensor_names=True)  # bool 是否打印所有的tensor的name
