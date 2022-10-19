import functions as fn
import model_NLP as model

DEBUG = True

df = fn.load_dataset(['eng','spa'],DEBUG)

df, cls_to_num, num_to_cls = fn.pre_encoding(df,DEBUG)

X_train,X_test,y_train,y_test = fn.split_training_testing(df,DEBUG)

train_seq, test_seq, tok = model.tokenize_and_sequence(X_train,X_test,DEBUG=DEBUG)

# build model

#test evaluate model

# save model

# loadl model

# predictions


