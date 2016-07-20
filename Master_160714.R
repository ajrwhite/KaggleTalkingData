# TalkingData Mobile


# Scoring function: Multi-class logarithmic loss --------------------------

mc_logloss = function(prediction_matrix, actual_class) {
  n = length(actual_class)
  if(nrow(prediction_matrix) != n) stop("y and y-hat different sizes")
  classes = c('F23-', 'F24-26','F27-28','F29-32', 'F33-42', 'F43+', 
              'M22-', 'M23-26', 'M27-28', 'M29-31', 'M32-38', 'M39+')
  k = length(classes)
  # Convert actual_class vector to a matrix representation
  actual_matrix = matrix(rep(0, n*k), nrow=n, ncol=k)
  colnames(actual_matrix) = classes
  for (i in 1:n) {
    actual_matrix[i,actual_class[i]] = 1
  }
  return(-((sum(rowSums(actual_matrix * log(prediction_matrix))))/n))
}

# Load Libraries ----------------------------------------------------------

library(data.table)
library(ggplot2)
library(bit64)
library(Matrix)


# Load data ---------------------------------------------------------------

app_events = fread("Data\\app_events.csv")
ga_train = fread("Data\\gender_age_train.csv")
ga_test = fread("Data\\gender_age_test.csv")
phone = fread("Data\\phone_brand_device_model.csv")
events = fread("Data\\events.csv")
app_labels = fread("Data\\app_labels.csv")
label_cats = fread("Data\\label_categories.csv")

# Tidy data ---------------------------------------------------------------

setkeyv(app_events, c("event_id", "app_id"))
setkey(ga_train, "device_id")
setkey(ga_test, "device_id")
setkey(phone, "device_id")
setkey(events, "event_id")
setkeyv(app_labels, c("app_id", "label_id"))
setkey(label_cats, "label_id")

# Join sets ----------------------------------------------

train = merge(ga_train, phone)
train = train[which(duplicated(train)==FALSE),]
test = merge(ga_test, phone)
test = test[which(duplicated(test)==FALSE),]

train_y = copy(ga_train)

train[,tt := "train"]
test[,tt := "test"]

train[,gender := NULL]
train[,age := NULL]
train[,group := NULL]

all_x = rbind(train, test)
rm(list=c("train", "test"))


# Add dummy variables for key brands --------------------------------------
all_x[,phone_brand := paste0("brand_", as.numeric(as.factor(phone_brand)))]
all_x[,device_model := paste0("device_", as.numeric(as.factor(device_model)))]
# Add dummy brands
train_brands = table(all_x[tt=="train",phone_brand])
test_brands = table(all_x[tt=="test",phone_brand])
train_brands = train_brands[train_brands > 100]
test_brands = test_brands[test_brands > 100]
top_brands = names(train_brands)[names(train_brands) %in% names(test_brands)]
for (b in top_brands) {
  all_x[,(b) := as.numeric(phone_brand == b)]
}
# Add dummy models
train_models = table(all_x[tt=="train",device_model])
test_models = table(all_x[tt=="test",device_model])
train_models = train_models[train_models > 100]
test_models = test_models[test_models > 100]
top_models = names(train_models)[names(train_models) %in% names(test_models)]
for (m in top_models) {
  all_x[,(m) := as.numeric(device_model == m)]
}

# Add geographical summary data -------------------------------------------
gvars = c("ave_lat", "ave_long", "var_lat", "var_long")
events[longitude==0, longitude := NA]
events[latitude==0, latitude := NA]
# ************** REWRITE ALL OF THIS TO USE DATA.TABLE AGGREGATE WHEN YOU HAVE THE TIME *********
ave_lat = as.data.table(aggregate(events[,latitude], list(events[,device_id]), function(x) mean(x, na.rm=T)))
ave_long = as.data.table(aggregate(events[,longitude], list(events[,device_id]), function(x) mean(x, na.rm=T)))
var_lat = as.data.table(aggregate(events[,latitude], list(events[,device_id]), function(x) var(x, na.rm=T)))
var_long = as.data.table(aggregate(events[,longitude], list(events[,device_id]), function(x) var(x, na.rm=T)))
names(ave_lat) = c("device_id", gvars[1])
names(ave_long) = c("device_id", gvars[2])
names(var_lat) = c("device_id", gvars[3])
names(var_long) = c("device_id", gvars[4])
setkey(ave_lat, "device_id")
setkey(ave_long, "device_id")
setkey(var_lat, "device_id")
setkey(var_long, "device_id")
all_x = merge(all_x, ave_lat, "device_id", all.x=T)
all_x = merge(all_x, ave_long, "device_id", all.x=T)
all_x = merge(all_x, var_lat, "device_id", all.x=T)
all_x = merge(all_x, var_long, "device_id", all.x=T)
for (v in gvars) {
  all_x[is.nan(get(v)), (v) := NA]
}
rm(list=gvars)


# Add app usage statistics ------------------------------------------------

num_events = as.data.table(aggregate(events[,latitude], list(events[,device_id]), length))
names(num_events) = c("device_id", "num_events")
setkey(num_events, "device_id")
all_x = merge(all_x, num_events, "device_id", all.x=T)
rm(num_events)

# Check for usage of top 200 apps
app_events[,app_id_numeric := paste0("app_", as.numeric(as.factor(app_id)))]
top_app_tab = table(app_events[,app_id_numeric])
top_app_tab = tail(sort(top_app_tab), 200)
events = merge(events, subset(app_events, select=c("event_id", "app_id_numeric")), "event_id", all.x=T)
rm(list=c("app_events", "app_labels", "label_cats", "phone"))
for (a in names(top_app_tab)) {
  events[,(a) := sum(app_id_numeric==a, na.rm=T), by = device_id]
  all_x = merge(all_x, unique(subset(events, select=c("device_id", a))), "device_id", all.x=T)
  events[,(a) := NULL]
}
for (a in names(top_app_tab)) {
  all_x[,paste0(a, "_freq") := get(a) / num_events]
}

# Timestamp analysis
# NEED TO WORK ON SOMETHING MEMORY WISE.

# Build XGB model to predict gender
library(xgboost)
modelvars = setdiff(names(all_x), c("device_id", "phone_brand", "device_model", "tt"))
train_x = as.matrix(subset(all_x, select=modelvars, subset=tt=="train"))
test_x = as.matrix(subset(all_x, select=modelvars, subset=tt=="test"))
dtrain_gender = xgb.DMatrix(train_x, label = as.numeric(ga_train[,gender]=="M"), missing=NA)
dtrain_group = xgb.DMatrix(train_x, label = as.numeric(as.factor(ga_train[,group]))-1, missing=NA)
dtest = xgb.DMatrix(test_x, missing=NA)
params_gender = list(objective = "binary:logistic", eta = 0.05,
             max_depth = 11, subsample = 0.75, colsample_bytree=0.5,
             metrics = "logloss")
params_group = list(objective = "multi:softprob", eta=0.05,
                    max_depth=11, subsample=0.75, colsample_bytree=0.5,
                    metrics="logloss")
set.seed(1984)
# Gender
cv_gender = xgb.cv(nfold = 4, print = 10,
            nrounds = 200, params = params_gender, data = dtrain_gender, nthread=3)
# Group
cv_group = xgb.cv(nfold=4, print=10, eval_metric="mlogloss",
                  nrounds=200, params=params_group, data=dtrain_group, nthread=3, num_class=12)
xgb_group = xgboost(print=10, eval_metric="mlogloss",
                    nrounds=200, params=params_group, data=dtrain_group, nthread=3, num_class=12)