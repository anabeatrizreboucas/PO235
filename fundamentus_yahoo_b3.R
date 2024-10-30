# Fundamentus + Yahoo Finance
# Ana Beatriz Rebouças
# Oct 28, 2024

rm(list = ls())

options(scipen = 999)
options(digits = 6)

library(rvest)
library(stringr)
library(tidyverse)
library(quantmod)

wd <- "/Users/anabeatrizreboucas/Documents/ITA/PO-235/Projeto"
setwd(wd)

# 1. Collect data from Fundamentus ####
url <- "https://www.fundamentus.com.br/resultado.php"

html <- read_html(url)

# Store all data provided by "resultado": one single string with all data ####
fund0 <- html_text(html_nodes(html, ".resultado"))

# 2. Organize Fundamentus data ####

 ## Split data (split elements considering "\n" as the separator) ####
fund0 <- str_split(fund0, "\n")

## we have a list with only 1 element. Save as a vector ####
fund0 <- fund0[[1]]

## remove blank spaces ####
fund0 <- str_remove_all(fund0, " ")
fund0 <- str_remove_all(fund0, "\t")

## Convert to a dataframe (21 variables) ####
fund1 <- as.data.frame(matrix(fund0, ncol = 21,  byrow = TRUE), 
                         stringsAsFactors = FALSE)

## Last row repeats the header. Remove it ####
fund1[nrow(fund1),]
fund2 <- fund1[-nrow(fund1),]

## Names of variables are stored in the first row. Adopt as the header and remove first row ####
names(fund2) <- fund2[1,]
fund3 <- fund2[-1,]

## Change thousand separators (from "." to "") and decimal (from "," to ".") ####
fund4 <- fund3 %>%
  mutate_all(~ str_replace_all(., "\\.", "")) %>%
  mutate_all(~ str_replace(., ",", "."))

## List of variables extracted from Fundamentus ####
    # PSL = Preço sobre lucro
    # PSVP = Preço sobre Valor Patrimonial
    # PSR = Preço sobre receita líquida
    # div_yield = Dividendo sobre preço da ação
    # PSA = Preço sobre ativos
    # PSCG = Preço sobre capital de giro
    # PSEBIT = Preço sobre EBIT
    # PSACL = Preço sobre ativo circulante líquido
    # EV_EBIT = EV/EBIT
    # EV_EBITDA = EB/EBITDA
    # margem_EBIT = EBIT/RL
    # margem_liq = margem líquida
    # liq_corrente = ativo circulante / passivo circulante
    # ROIC = retorno sobre o capital empregado
    # ROE = retorno sobre o PL
    # liq_2meses = volume diário médio negociado nos ultimos 2 meses
    # PL = patrimonio líquido
    # div_bruta_PL = divida bruta sobre patrimônio líquido
    # taxa_cresc_RL_5anos = taxa de crescimento da receita líquida (últimos 5 anos)

## Renaming the variables ####
names(fund4) <- c("papel", "cotacao", "PSL",
                    "PSVP", "PSR", "div_yield",
                    "PSA", "PSCG", "PSEBIT",
                    "PSACL", "EV_EBIT", "EV_EBITDA", 
                    "margem_EBIT", "margem_liq", "liq_corrente",
                    "ROIC", "ROE", "liq_2meses",
                    "PL", "div_bruta_PL", "taxa_cresc_RL_5anos")

## Detect variables with values in percentage ####
   # These variables should be divided by 100 after being converted to numeric
summary_p <- fund4 %>% 
  summarize_all(~unique(str_detect(., "%"))) %>% t() 

summary_p <- tibble::rownames_to_column(as.data.frame(summary_p), "variable") %>% 
  rename(value = V1)

summary_p

  # Variables that are in percentage
vars_p <- summary_p$variable[which(summary_p$value == TRUE)]

  # Other numeric variables
vars_n <- summary_p$variable[which(summary_p$value == FALSE & summary_p$variable != "papel")] 

## Converting variables to numeric ####
fund5 <- fund4 %>%
  mutate(across(any_of(vars_p), ~as.numeric(str_remove(., "%"))/100),
         across(any_of(vars_n), as.numeric))

# 3. Checking if the data is in the tidy format ####

# Check case with more than 1 observation per "papel" ####
summary_fund <- fund5 %>%
  group_by(papel) %>%
  summarize(n = n()) %>% arrange(desc(n)) %>% filter(n > 1)

summary_fund

fund5 %>% filter(papel %in% summary_fund$papel)
  # Duplicates. We can simply select the unique values.

fund6 <- fund5 %>% unique()

head(fund6)

# 3.1. Add "papel" description ####

descr_acoes <- html |> html_elements(".tips") 

descr_acoes

acoes <- data.frame("papel" =  descr_acoes |> html_elements("a") |> html_text2(),
                    "descricao_papel" = descr_acoes |> html_attr("title")) %>% arrange(papel)

acoes <- unique(acoes)

head(acoes)

fund <- fund6 %>%
  left_join(acoes) %>%
  select(papel, descricao_papel, everything())

# 3.2. Save fundamentus data ####
  
  # Include date information given that Fundamentus does not provide historical data
  # and the table updates daily.

date <- format(Sys.Date(), "%Y%m%d")

write.csv(fund, paste0(wd,"/fundamentus_data/fundamentus_", date, ".csv"))


# 4. Define stocks to be analyzed (filters) ####
  
  stock_sel_table <- fund %>%
  filter(liq_2meses >= 1000000) # Liquidez > 1M

  stock_sel <- stock_sel_table$papel

# 5. Load historical stocks data from Yahoo Finance ####

 ## 5.1. Load stock data from quantmod package (time series)
    # getSymbols function from quantmod package provides stock information for a given stock 
    # ("Symbol") and time interval. It's possible to request more than one symbol at time, but
    # it would create individual elements for each stock. I preferred to make individual requests
    # for each stock, store the results in a list, and then join them all in a unique table.

  list_aux <- list()
  
  for(i in 1:length(stock_sel)){
    tryCatch({
      stock_aux <- stock_sel[i]
      df_aux <- getSymbols(paste0(stock_aux, ".SA"), src = "yahoo",
                           from = as.Date('2022-01-01'), auto.assign = FALSE)
      
      df_aux <- as.data.frame(df_aux) %>%
        mutate(date = row.names(.))
      
      row.names(df_aux) <- NULL
      
      list_aux[[i]] <- df_aux %>%
        select(date, everything())
    }, error = function(e){cat("ERROR :",conditionMessage(e), "\n")})
  }
  
 ## 5.2. Removing NULL entries (errors from the previous loop)  

  list_aux <- Filter(Negate(is.null), list_aux) 
  
 # Removing stocks listed in "stocks_sel" but not found in our data request for yahoo.
 # These stocks are detailed below (section 6.1).

  ## 5.3. Understanding the data structure
    # Each element of the list (data of each stock) will have specific column names,
    # based on their stock name. For example:  
  
  head(list_aux[[1]]) 
  head(list_aux[[2]]) 
  
  # The prefix of the column name is the stock name and the suffix is the variable name 
  # ("Open", "High", "Low", "Close", "Volume". and "Adjusted"). 
  
  
  ## 5.4. Joining all elements (all stocks) in one table 
  df <- list_aux %>%
    reduce(full_join, by = "date")

  # This data is not tidy yet. Stocks data is distributed across columns and not rows. 
  # Below an extract of the dataset is presented (first 6 rows and first 8 columns).  
  head(df)[,1:8]
  
  ## 6. Making the data tidy ####
  df0 <- df %>%
    pivot_longer(cols = -date,
                 names_to = "variable",
                 values_to = "value",
                 values_drop_na = TRUE)
  
  head(df0)

   # Not tidy yet. All variables are stored in one column named "variable" and their value
   # is stored in a column named "value". This column also contains the stock name. I 
   # separate this information into 2 variables: 
      ### "stock": the stock name, and
      ### "var": variable name  
  
  df0 <- df0 %>%
    mutate(stock = str_extract(variable,".*(?=\\.SA)"),
           var = str_extract(variable, "(?<=SA\\.).*"))
  
  head(df0)

 df1 <- df0 %>% select(-variable) %>%
    pivot_wider(names_from = var,
                values_from = value) %>%
    select(stock, date, everything()) %>%
    arrange(stock, date)
  
  head(df1)  

  # df1 is now tidy.
 
   write.csv(df1, file = paste0(wd,"/yahoo_data/yahoo_b3_", date, ".csv"), row.names = FALSE)  
