if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)

# data import from GitHub
dat15 <- read.csv("https://raw.githubusercontent.com/Airborne737/World_Happiness/master/2015.csv")
dat19 <- read.csv("https://raw.githubusercontent.com/Airborne737/World_Happiness/master/2019.csv")

# column renaming and matching
dat15 <- dat15 %>% 
  rename(country = Country, score = Happiness.Score, GDP_capita = Economy..GDP.per.Capita., healthy_life_expectancy = Health..Life.Expectancy., freedom = Freedom, generosity = Generosity, corruption = Trust..Government.Corruption.) %>%
  select(country, score, GDP_capita, healthy_life_expectancy, freedom, generosity, corruption)

dat19 <- dat19 %>%
  rename(country = Country.or.region, score = Score, GDP_capita = GDP.per.capita, healthy_life_expectancy = Healthy.life.expectancy, freedom = Freedom.to.make.life.choices, generosity = Generosity, corruption = Perceptions.of.corruption) %>%
  select(country, score, GDP_capita, healthy_life_expectancy, freedom, generosity, corruption)

# create training and testing sets
set.seed(1, sample.kind="Rounding")
test_index <- createDataPartition(y = dat15$score, times = 1, p = 0.5, list = FALSE)
train_set <- dat15[-test_index,]
test_set <- dat15[test_index,]

# RMSE defined
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

# exploratory data analysis
# str
str(dat15)
str(dat19)

# summary statistics
summary(dat15)
summary(dat19)

# happiest countries 2015
dat15 %>%
  arrange(-score) %>%
  top_n(20, score) %>%
  ggplot(aes(score, reorder(country, score))) +
  geom_bar(color = "black", fill = "forestgreen", stat = "identity") +
  xlab("Happiness Score") +
  ylab(NULL) +
  theme_bw()

# happiest countries 2019
dat19 %>%
  arrange(-score) %>%
  top_n(20, score) %>%
  ggplot(aes(score, reorder(country, score))) +
  geom_bar(color = "black", fill = "deepskyblue2", stat = "identity") +
  xlab("Happiness Score") +
  ylab(NULL) +
  theme_bw()

# distribution of happiness scores
dat19 %>%
  ggplot(aes(score)) +
  geom_histogram(color = "black", fill = "deepskyblue2", bins = 30) +
  labs(x = "Happiness Scores", y = "Count") +
  scale_x_continuous() +
  theme_bw()

# correlation matrix
dat19 %>%
  select(-country) %>%
  cor()

# countries with the highest GDP per capita
dat19 %>%
  arrange(-GDP_capita) %>%
  top_n(20, score) %>%
  ggplot(aes(GDP_capita, reorder(country, GDP_capita))) +
  geom_bar(color = "black", fill = "deepskyblue2", stat = "identity") +
  xlab("GDP per Capita") +
  ylab(NULL) +
  theme_bw()

# model 1 lm
lm_train <- train_set %>% select(-country) %>% train(score ~ ., method = "lm", data = .)
lm_predict <- predict(lm_train, test_set)
lm_result <- RMSE(test_set$score, lm_predict)
results <- tibble(Method = "Model 1: Linear regression", RMSE = lm_result)
results %>% knitr::kable()

# model 2 rf
set.seed(1, sample.kind="Rounding")
fitcontrol <- trainControl(method = "repeatedcv", number = 10, repeats = 3)
rf_train <- train_set %>% select(-country) %>% train(score ~ ., method = "rf", trControl = fitcontrol, data = .)
rf_predict <- predict(rf_train, test_set)
rf_result <- RMSE(test_set$score, rf_predict)
results <- bind_rows(results, tibble(Method = "Model 2: Random forest", RMSE = rf_result))
results %>% knitr::kable()

# model 3 ranger
set.seed(1, sample.kind="Rounding")
ranger_train <- train_set %>% select(-country) %>% train(score ~ ., method = "ranger", trControl = trainControl(method = "cv", number = 5), num.trees = 500, data = .)
ranger_predict <- predict(ranger_train, test_set)
ranger_result <- RMSE(test_set$score, ranger_predict)
results <- bind_rows(results, tibble(Method = "Model 3: Ranger RF", RMSE = ranger_result))
results %>% knitr::kable()

# model 4 knn
set.seed(1, sample.kind="Rounding")
knn_train <- train_set %>% select(-country) %>% train(score ~ ., method = "knn", trControl = trainControl(method = "repeatedcv", number = 10, repeats = 3), data = .)
knn_predict <- predict(knn_train, test_set)
knn_result <- RMSE(test_set$score, knn_predict)
results <- bind_rows(results, tibble(Method = "Model 4: K-Nearest Neighbors", RMSE = knn_result))
results %>% knitr::kable()

# train knn on dat15 and final validation on dat19
set.seed(1, sample.kind="Rounding")
knn_train15 <- dat15 %>% select(-country) %>% train(score ~ ., method = "knn", trControl = trainControl(method = "repeatedcv", number = 10, repeats = 3), data = .)
knn_predict19 <- predict(knn_train15, dat19)
final_result <- RMSE(dat19$score, knn_predict19)
results <- bind_rows(results, tibble(Method = "Final validation: K-Nearest Neighbors", RMSE = final_result))
results %>% knitr::kable()

