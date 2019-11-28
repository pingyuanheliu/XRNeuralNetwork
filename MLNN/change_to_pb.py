import tensorflow.compat.v1 as tf
from nets.mobilenet import MobileNetV2 
def inference(input_tf,n_classes): 
    net = MobileNetV2(n_classes=n_classes,depth_rate=1.0,is_training=False) 
    output = net.build_graph(input_tf) 
    return output
  
def main(): 
    graph = tf.Graph()  
    
    with graph.as_default():
        input_tf = tf.placeholder(tf.float32,shape=(64,64,3))
        #logits
        output_tf = inference(tf.expand_dims(input_tf,axis=0), 3755)
        output_tf = tf.nn.softmax(output_tf) 
        values, indices = tf.nn.top_k(output_tf, k=5, sorted=True, name=None)
        #加载模型参数
        print('loading model')
        saver = tf.train.Saver()
    
    with tf.Session(graph=graph) as sess:
        
        saver.restore(sess,'model/model-37700') 
         
        print('Exporting trained model to', 'model_for_serving')
        builder = tf.saved_model.builder.SavedModelBuilder('model_for_serving')
        #Tensor对象转TensorInfo对象
        tensor_info_x = tf.saved_model.utils.build_tensor_info(input_tf)
        tensor_info_p = tf.saved_model.utils.build_tensor_info(tf.squeeze(values, axis=0)) 
        tensor_info_c = tf.saved_model.utils.build_tensor_info(tf.squeeze(indices, axis=0)) 
        #定义SignatureDef对象
        prediction_signature =  tf.saved_model.signature_def_utils.build_signature_def(
                    inputs={'images': tensor_info_x},
                    outputs={'scores': tensor_info_p,'classes':tensor_info_c},
                    method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME)
        #为不同的SignatureDef对象映射名称
        DEFAULT_KEY = tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
        signature_map = {'predict_images': prediction_signature,
                          DEFAULT_KEY:prediction_signature}
        #读取加入图和变量
        builder.add_meta_graph_and_variables(sess, 
                                             [tf.saved_model.tag_constants.SERVING],
                                             signature_def_map = signature_map,
                                             main_op=tf.tables_initializer(),
                                             strip_default_attrs=True)
        #保存为Tensorflow Serving能识别的模型
        builder.save()
    
        print('Done!')
        pass
    pass
if __name__=='__main__':
    main()