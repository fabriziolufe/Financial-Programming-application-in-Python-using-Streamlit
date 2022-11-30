# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 15:46:36 2022

@author: flucerofernandez
"""



# FINANCIAL DASHBOARD #2 - v2.1
###############################################################################

#==============================================================================
# Initiating
#==============================================================================

import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import datetime
import yfinance as yf
import streamlit as st
from streamlit_option_menu import option_menu
import pandas_datareader.data as web
import numpy as np
import tweepy
from textblob import TextBlob
import time
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from st_aggrid import AgGrid

#First step, we want the app to be shown as wide format. For better Visualization.
st. set_page_config(layout="wide")

#The entire code design template of creating tabs as functions comes from our Professor Minh Phan work in class
# link to template code: https://drive.google.com/file/d/1Se7Zl07oEuXo4BI-mZhhI6B_KRSWjBo4/view?usp=sharing

#==============================================================================
# Tab 1 Summary
# Code includes the creation of the following: 
#1. Collect and present the information of the stock similar to the tab Yahoo Finance > Summary
#2. In the chart, there should be an option to select different duration of time  i.e. 1M, 3M, 6M, 
#3. Present also the company profile, description and major shareholders
#==============================================================================

def tab1():
    
    # Add dashboard title and description
    st.title("Financial Dashboard with Streamlit MBD 2022")
    st.write("Data source: Yahoo Finance. URL: https://finance.yahoo.com/")
    st.write("Github from Creator URL: https://github.com/fabriziolufe")
    st.header('Summary Information')
    
    #Creating the selection box that will have all possible time periods for the graph visualization.
    select_period_tab1 = st.selectbox("Select time period", ['Selected Range','1mo','3mo','6mo','1y','3y','5y','ytd','max'])
    
    
    @st.cache
    #Function to get ticker info, as key statistics, summary information of the company. 
    #followed documentation from: https://pypi.org/project/yfinance/ 
    def GetCompanyInfo(ticker):
        return yf.Ticker(ticker).info
    
    if ticker != '':
        # Get the company information in list format
        info = GetCompanyInfo(ticker)
        
        GetCompanyInfo(ticker)
        
        # Show the company description
        st.write('**1. Business Summary:**')
        
        #create columns in order to centrate photo in the middle 
        col1, col2, col3 = st.columns([10,6,10])

        with col1:
            #blank
                st.write("   ")

        with col2:
            #show the logo of the company selected.
            st.image(info['logo_url'], width=200, output_format="JPG")

        with col3:
            #blank
            st.write("   ")
       
        #call info variable and retrieve the Long Business Summary of the selected company.
        st.write(info['longBusinessSummary'])
        
       
        #create columns in order to put info beside each other and not below. 
        col1, col2, col3 = st.columns([6,6,12])

        with col1:
          # Show some statistics that can be retrieved from .info
          st.write('**2. Key Statistics:**')
          #create keys for your dict.
          keys = ['previousClose', 'open', 'bid', 'ask', 'marketCap','enterpriseValue',
                  'volume', 'averageVolume','beta', 'trailingEps']
          company_stats = {}  # Dictionary
          for key in keys:
              #connect the keys with their actual values in .info
              company_stats.update({key:info[key]})
         #convert to a dataframe (easier manipulation)
          company_stats = pd.DataFrame({'Value':pd.Series(company_stats)}).reset_index().rename(
              columns={'index':'Indicators for ' +ticker})  # change column name
          #use AgGrid (library) for a nicer looking table in streamlit.
          AgGrid(company_stats)
          
          
        with col2:
            #showing major holders. with . major_holders
            st.write('**3. Shareholders:**')
            #convert to a Dataframe and rename for handling data better
            holders=pd.DataFrame(yf.Ticker(ticker).major_holders).rename(columns={0:'shares', 1:'holders'})
            #create a for loop to iterate inside the holders dataframe and retrieve into an st metric the name and value. 
            for i in range(len(holders)):
              st.metric(label=holders['holders'][i], value=holders['shares'][i]) 
    
        #depending on the user decision to select specific dates or not, .history will be different.
        if select_period_tab1 == 'Selected Range':
            if show_dates:
                #using the dates the user provided for analysis
                stock_price= yf.Ticker(ticker).history(start=start_date, end=end_date, interval='1d').reset_index()  
                #if the option is selected but the user has not provided dates, then by default analyze 60 days.
            else: stock_price= yf.Ticker(ticker).history(start_date=str(datetime.datetime.today().date() - datetime.timedelta(days=60)), 
                                 end_date=datetime.datetime.today(), interval='1d').reset_index() 
            
        else: #if user decides a diferrent period, modify .history accordingly.
            stock_price=yf.Ticker(ticker).history(period=select_period_tab1, interval='1d').reset_index() 
             
            # Add an area plot with plotly- Link to documentation: https://plotly.com/python/filled-area-plots/
        with col3:
            # since we want to change the color of the graph depending if the stock is "up or "down"
            #understanding the stock charts: https://bullishbears.com/red-to-green-move-stocks/
            #we will create a new column that saves the color depending if Close price is more than Open. 
            stock_price['color']=np.where(stock_price['Close']>stock_price['Open'],'green','red')
            #We now count the number of values for each case and sort to have the winner first as index 0. 
            count_color_values=stock_price['color'].value_counts().reset_index().sort_values(by='color',ascending=False)
            #secondary_y to create two graphs in one. 
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            #key -> marker_color=df['color'] in the volume Bars
            fig.add_trace(go.Bar(name='Volume of shares',x=stock_price['Date'], y = stock_price['Volume'],
                                 marker_color=stock_price['color']))
            #fill tozeroy will fill with color the scatter chart. 
            fig.add_trace(go.Scatter(name='Close price of shares',x=stock_price['Date'],y= stock_price['Close'],
                                     marker_color=count_color_values['index'][0], fill='tozeroy'),secondary_y=True)
            #change axis length to be similar to yahoo finance.
            fig.update_layout(title= 'Shares for '+ticker,  xaxis_rangeslider_visible=True, 
                              autosize=True, 
                              yaxis1 = dict(range=[0, max(stock_price['Volume'])*1.3]),
                              yaxis2 = dict(range=[0, max(stock_price['Close'])+20]))   
            #visualize in streamlit plotly charts.
            st.plotly_chart(fig)

#==============================================================================
# Tab 2 TIME ANALYSIS 
#1. Collect and present the stock price similar to the tab Yahoo Finance chart.
#2. Select the date range for showing the stock price.
#3. Show the stock price for different duration of time i.e. 1M, 3M, 
#4. Switch between different time intervals i.e. Day, Month, Year. 
#5. Switch between line plot and candle plot for the stock price.
#6. Show the trading volume at the bottom of the chart.
#7. Show the simple moving average (MA) for the stock price using a window size of 50 days
#==============================================================================

def tab2():
    
    # Add dashboard title and description
    st.title("Financial Dashboard with Streamlit MBD 2022")
    st.write("Data source: Yahoo Finance. URL: https://finance.yahoo.com/")
    st.write("Github from Creator URL: https://github.com/fabriziolufe")
    st.header('Chart - Graphic Analysis')
    
    #This is the best feature of the app: FORMS. it let's you encapsulate parameters to be run just with a button.
    #Source: https://www.youtube.com/watch?v=FwvesOnxP60
    #create the form
    form_tab2=st.form(key='Lima')
    #create the options you want inside your form. such as the interval.
    select_interval= form_tab2.selectbox("Select time interval", ['1d','5d','1mo','3mo','6mo'])
    
    #use conditionals, in case the user selects certain intervals just show rational options. 
    if select_interval == '3mo':
        #creating another parameter time period.
        select_period = form_tab2.selectbox("Select time period", ['Selected Range','6mo','1y','3y','5y','ytd','max'])
     #creating another parameter time period.
    elif select_interval == '6m':
        select_period = form_tab2.selectbox("Select time period", ['Selected Range','1y','3y','5y','max']) 
     #creating another parameter time period.
    elif select_interval =='12m':
        select_period = form_tab2.selectbox("Select time period", ['Selected Range','3y','5y','max'])   
     #creating another parameter time period.
    else:
        select_period = form_tab2.selectbox("Select time period", ['Selected Range','1mo','3mo','6mo','ytd','1y','3y','5y','max']) 
    
    # Retrieve the stock data from the selected ticker.
    @st.cache
    #followed documentation from: https://pypi.org/project/yfinance/ 
    def GetStockData(tickers, start_date=str(datetime.datetime.today().date() - datetime.timedelta(days=60)), 
                     end_date=datetime.datetime.today()):
        stock_price = pd.DataFrame() # create dataframe
        
        # call the . history, parameters depending on users selection of period and interval.
        for tick in tickers:
            if select_period == 'Selected Range':
                stock_df = yf.Ticker(tick).history(start=start_date, end=end_date, interval=select_interval)
            else: 
                stock_df = yf.Ticker(tick).history(period=select_period, interval=select_interval)
            
            stock_df['Ticker'] = tick  # Add the column ticker name
            stock_price = pd.concat([stock_price, stock_df], axis=0)  # Comebine results
        return stock_price.loc[:, ['Ticker', 'Open', 'High', 'Low', 'Close', 'Volume']]
   
    # Add a check box this will be inside the form also. 
    show_data = form_tab2.checkbox("Show me the money. I want to see the data!")
    if show_dates:
        if ticker != '':
            stock_price = GetStockData([ticker], start_date, end_date).reset_index()
            if show_data:
                st.write('Stock price data')
                st.table(stock_price)
    else: # by default show 60 days of data
        if ticker != '':
            stock_price = GetStockData([ticker], start_date=str(datetime.datetime.today().date() - datetime.timedelta(days=60)), 
                             end_date=datetime.datetime.today()).reset_index()
            if show_data:
                st.write('Stock price data')
                st.table(stock_price)
     
    #Create a different .history that will call the data with period='max' and calculate the moving average
    #how to calculate moving average:https://www.geeksforgeeks.org/how-to-calculate-moving-averages-in-python/             
    M_Average= pd.DataFrame(yf.Ticker(ticker).history(period='max', interval=select_interval)).reset_index()
    #50 days
    M_Average['M50']=M_Average['Close'].rolling(50).mean()
    #filter with our selected dates.
    M_Average=M_Average[M_Average['Date'].isin(stock_price['Date'])]

            
    # Add plots with plotly
    if ticker != '':
        
        
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        #also inside form from tab2 a radio with the chart type.
        select_chart_type=form_tab2.radio('Select type of chart',['Line','Candle'], horizontal = True)
        
        #button that will initiate all commands inside the tab.
        form_tab2.form_submit_button('Show results')
        
        #same color logic as in tab1.
        st.header('**Market results for '+ticker+' from Yahoo Finance**')
        stock_price['color']=np.where(stock_price['Close']>stock_price['Open'],'darkolivegreen','crimson')
        count_color_values=stock_price['color'].value_counts().reset_index().sort_values(by='color',ascending=False)
        
        if select_chart_type =='Line':
            fig.add_trace(go.Scatter(name= 'Close Value',x=stock_price['Date'],
                                        y=stock_price['Close'], marker_color=count_color_values['index'][0]),
                          secondary_y=True)
            fig.add_trace(go.Bar(name='Volume of shares',x=stock_price['Date'], y = stock_price['Volume'], 
                                 marker_color=stock_price['color']))
            #extra trace different from first chart, includes Moving Average, 
            fig.add_trace(go.Scatter(name='Moving Average/ 50 days',x=M_Average['Date'], y = M_Average['M50'],
                                     marker_color='turquoise'),secondary_y=True)
            
            fig.update_layout(xaxis_rangeslider_visible=True, autosize=True,
                              yaxis1=dict(range=[0, max(stock_price['Volume'])*1.3])) 
            #modify width.                     
            st.plotly_chart(fig, use_container_width=True)
        
        elif select_chart_type =='Candle':
            fig.add_trace(go.Candlestick(name= 'Stock value (Open, High, Low, Close)',x=stock_price['Date'],
                                         open= stock_price['Open'], 
                                         high= stock_price['High'],
                                         low= stock_price['Low'],
                                         close= stock_price['Close']), secondary_y=True)
        
            fig.add_trace(go.Bar(name='Volume of shares',x=stock_price['Date'], y = stock_price['Volume'],
                                 marker_color=stock_price['color']))
            
            fig.add_trace(go.Scatter(name='Moving Average/ 50 days',x=M_Average['Date'], y = M_Average['M50'],
                                     marker_color='turquoise'),secondary_y=True)
            
            
            fig.update_layout(  xaxis_rangeslider_visible=True, autosize=True,
                              yaxis1=dict(range=[0, max(stock_price['Volume'])*1.3])) 
                                     
            st.plotly_chart(fig, use_container_width=True)
            

#==============================================================================
# Tab 3 FINANCIAL STATEMENTS.
#1. Collect and present the financial information of the stock similar to the tab Yahoo Finance > Financials.
#2. There should be an option to select between Income Statement, Balance Sheet and Cash Flow.
#3. There should be an option to select between Annual and Quarterly period
#==============================================================================
def tab3():
    
        # Add dashboard title and description
        st.title("Financial Dashboard with Streamlit MBD 2022")
        st.write("Data source: Yahoo Finance. URL: https://finance.yahoo.com/")
        st.write("Github from Creator URL: https://github.com/fabriziolufe")
        st.header('Financials')
        
        #columns to organize better the parameters
        col1, col2 = st.columns([15,15])
        form_financial= st.form(key='Lille')
        #inside the created form, a selectbox with the type of statement.
        with col1:
            select_financial= form_financial.selectbox("Select type of financial info", ['Income Statement','Balance Sheet','Cash Flow'])
        #and the time window of analysis
        with col2: 
            select_time_financial = form_financial.radio('',['Annual','Quarterly'], horizontal=True)
        # create button to show results.
        form_financial.form_submit_button('Show statements')
        #function for retrieving financials.
        @st.cache
        #followed documentation from: https://pypi.org/project/yfinance/ 
        def GetCompanyFinancials (ticker):
            #different calls for annual and quarterly info.
            if select_financial=='Balance Sheet':
                if select_time_financial == 'Annual':
                    fin_result=pd.DataFrame(yf.Ticker(ticker).balance_sheet)
                else: fin_result=pd.DataFrame(yf.Ticker(ticker).quarterly_balance_sheet)   
                
            elif select_financial=='Income Statement':
                if select_time_financial == 'Annual':
                    fin_result=pd.DataFrame(yf.Ticker(ticker).financials)#.reset_index().reindex([6,20,14,23,24,26,8,13,19,0,11,16,2,15,27,17,5,25,10,3,7,1,9,18,4,22,12,21]).set_index('index')
                else: fin_result=pd.DataFrame(yf.Ticker(ticker).quarterly_financials)#.reset_index().reindex([6,20,14,23,24,26,8,13,19,0,11,16,2,15,27,17,5,25,10,3,7,1,9,18,4,22,12,21]).set_index('index')   
            else: 
                if select_time_financial == 'Annual':
                    fin_result=pd.DataFrame(yf.Ticker(ticker).cashflow)
                else: fin_result=pd.DataFrame(yf.Ticker(ticker).quarterly_cashflow)
            return fin_result
        #fill nulls with $0. change format of columns and restyle the format of cells.
        #Source: https://www.geeksforgeeks.org/how-to-format-date-using-strftime-in-python/
        #Source: https://stackoverflow.com/questions/60610101/string-formatting-dollar-sign-in-python
        financial= GetCompanyFinancials(ticker).fillna(0).rename(
            lambda t: 'Up to: '+t.strftime('%d-%B-%Y'),axis='columns').style.format("${:0,.2f}")
        
        #show the name of the selected info and the table     
        st.header(select_time_financial+' '+select_financial)
        st.table(financial)


#==============================================================================
# Tab 4 MONTECARLO SIMULATION
#1. Conduct and present a Monte Carlo simulation for the stock closing price in the next n-days from today.
#2. The number of simulations should be selected from a dropdown list of n = 200, 500, 1000 simulations.
#3. The time horizon should be selected from a dropdown list of t = 30, 60, 90days from today.
#4. Estimate and present the Value at Risk (VaR) at 95% confidence interval.

#==============================================================================
        

 # Inspiration from Phan Minh code from course. 
 #Source Code: https://drive.google.com/file/d/1we38K8uu4QWZ9iGaHZmrTOHk9pxQRB_P/view?usp=sharing
 
def tab4():
    
        # Add dashboard title and description
        st.title("Financial Dashboard with StreamlitL MBD 2022")
        st.write("Data source: Yahoo Finance. URL: https://finance.yahoo.com/")
        st.write("Github from Creator URL: https://github.com/fabriziolufe")
        st.header('Monte Carlo Simulation')
        
        #new form for the tab.
        form_Montecarlo= st.form(key='Bogota')
        
        #organizing parameters.
        col1, col2 = form_Montecarlo.columns([10,10])
        with col1:
            #Creating parameters, N of simulations and time horizon ( to be connected to the simulation function)
            select_simulation=form_Montecarlo.radio('Select N of Simulations',['100','200','500','1000'], horizontal=True)
        with col2:
            select_horizon=form_Montecarlo.radio('Select time horizon',['30','60','90'], horizontal=True)
        #deploy button.
        form_Montecarlo.form_submit_button('Run simulation')
        
        #initiate st.progress button (it will charge with the number of simulations.)
        my_progress=st.progress(1)
        #Creating Montecarlo object.
        class MonteCarlo(object):
                        
                        def __init__(self, tickerstock, data_source, time_horizon, n_simulation, seed,
                                     start_date=str(datetime.datetime.today().date() - datetime.timedelta(days=60)), 
                                         end_date=datetime.datetime.today()):
                            
                            # Initiate class variables
                            self.tickerstock = tickerstock  # Stock ticker
                            self.data_source = data_source  # Source of data, e.g. 'yahoo'
                            self.start_date = datetime.datetime.strptime(str(start_date), '%Y-%m-%d')  # Text, YYYY-MM-DD
                            self.end_date = datetime.datetime.strptime(str(end_date), '%Y-%m-%d')  # Text, YYYY-MM-DD
                            self.time_horizon = time_horizon  # Days
                            self.n_simulation = n_simulation  # Number of simulations
                            self.seed = seed  # Random seed
                            self.simulation_df = pd.DataFrame()  # Table of results
                            
                            # Extract stock data
                            self.stock_price = web.DataReader(tickerstock, data_source, self.start_date, self.end_date)
                            
                            # Calculate financial metrics
                            # Daily return (of close price)
                            self.daily_return = self.stock_price['Close'].pct_change()
                            # Volatility (of close price)
                            self.daily_volatility = np.std(self.daily_return)
                        
                        @st.cache(suppress_st_warning=True)   
                        def run_simulation(self):
                            
                            # Run the simulation
                            np.random.seed(self.seed)
                            self.simulation_df = pd.DataFrame()  # Reset
                            
                            for i in range(self.n_simulation):
                    
                                # The list to store the next stock price
                                next_price = []
                    
                                # Create the next stock price
                                last_price = self.stock_price['Close'][-1]
                    
                                for j in range(self.time_horizon):
                                    
                                    # Generate the random percentage change around the mean (0) and std (daily_volatility)
                                    future_return = np.random.normal(0, self.daily_volatility)
                    
                                    # Generate the random future price
                                    future_price = last_price * (1 + future_return)
                    
                                    # Save the price and go next
                                    next_price.append(future_price)
                                    last_price = future_price
                 
                                # Store the result of the simulation
                                next_price_df = pd.Series(next_price).rename('sim' + str(i))
                                self.simulation_df = pd.concat([self.simulation_df, next_price_df], axis=1)
                                my_progress.progress(i/self.n_simulation)
                        
                        @st.cache
                        def plot_simulation_price(self):
                            
                            # Plot the simulation stock price in the future
                            fig, ax = plt.subplots()
                            fig.set_size_inches(15, 10, forward=True)
                    
                            plt.plot(self.simulation_df)
                            plt.title('Monte Carlo simulation for ' + self.tickerstock + \
                                      ' stock price in next ' + str(self.time_horizon) + ' days')
                            plt.xlabel('Day')
                            plt.ylabel('Price')
                    
                            plt.axhline(y=self.stock_price['Close'][-1], color='red')
                            plt.legend(['Current stock price is: ' + str(np.round(self.stock_price['Close'][-1], 2))])
                            ax.get_legend().legendHandles[0].set_color('red')
                            
                            return plt
                            
                        
                        def value_at_risk(self):
                            # Price at 95% confidence interval
                            future_price_95ci = np.percentile(self.simulation_df.iloc[-1:, :].values[0, ], 5)
                    
                            # Value at Risk
                            VaR = self.stock_price['Close'][-1] - future_price_95ci
                            return 'VaR at 95% confidence interval is: ' + str(np.round(VaR, 2)) + ' USD'   
        
        # Initiate depending on previous date parameters. by default is 60 days of analysis
        if show_dates:
            #start and end date will change if sidebar box is selected.
            mc_sim = MonteCarlo(tickerstock=ticker, data_source='yahoo',
                                start_date=start_date, end_date=end_date,
                                time_horizon=int(select_horizon), n_simulation=int(select_simulation), seed=123)
        else: 
            mc_sim = MonteCarlo(tickerstock=ticker, data_source='yahoo',
                                start_date=str(datetime.datetime.today().date() - datetime.timedelta(days=60)), 
                                    end_date=str(datetime.datetime.today().date()),
                                time_horizon=int(select_horizon), n_simulation=int(select_simulation), seed=123)
        
        @st.cache
        #create function to retrieve wanted functions from class montecarlo
        def building_results(class_mc):
            class_mc.run_simulation()
            return class_mc.plot_simulation_price(),class_mc.value_at_risk()
        st.set_option('deprecation.showPyplotGlobalUse', False)
        
        result=building_results(mc_sim)
        my_progress.empty()
       #show the simulation graph
        st.pyplot(result[0])
        # show the value at risk with 95% confidence interval.
        st.header(result[1])
            
#==============================================================================
#tab 5 NEWS SENTIMENT ANALYSIS
#1. Your own analysis
#2. You are free to design this tab with any financial analyses or additional information 
#==============================================================================
def tab5():

#SENTIMENT text mining analysis using TextBlob
#Source for inspiration only: https://www.youtube.com/watch?v=o_OZdbCzHUA&t=324s
#I believe this stock and news sentiment all together has not been done before (just used source to 
#learn how to call twitter tweets. )
# calling tweets from twitter - reference: https://www.youtube.com/watch?v=o_OZdbCzHUA&t=341s    
       st.title("Financial Dashboard with Streamlit MBD 2022")
       st.write("Data source: Yahoo Finance. URL: https://finance.yahoo.com/")
       st.write("Github from Creator URL: https://github.com/fabriziolufe")
       select_newstype= st.radio('Which news would you like to see?',['Regular news','Twitter messages'],horizontal=True)
       
       #organizing info - two types of news - Regular and Twitter.
       col1, col2=st.columns([15,15])
       with col1: 
           if select_newstype == 'Regular news':
               st.header('Media Sentiment Analysis of '+ticker)
               list_sentiment=[]
               #using yfinance .news https://pypi.org/project/yfinance/
               news=yf.Ticker(ticker).news
               for i in range(len(news)):
                    
                        st.caption(news[i]['publisher'])
                        #using TextBlob sentiment function to analyze 
                        #explanation:  https://www.geeksforgeeks.org/python-textblob-sentiment-method/
                        st.write(news[i]['title'])
                        sentiment=TextBlob(news[i]['title']).sentiment[0]
                        #append all sentiments into a list to calculate the mean after,
                        list_sentiment.append(sentiment)
               
            
           else:
               st.header('Twitter Sentiment Analysis of '+ticker)
               #information needed to call the tweets - from my developer account in twitter.
               twitter_key= 'TkQxdsyltu8uqb40lif6I2X2q'
               twitter_secret= 'ak0hQYjYDizzLdLKqV1GCAuPETvbZCotRI6GAWcXBJS3RTc9Xz'
               
               access= '3939707355-upOw4ZASz36RahUup8atZagpAKUxhAjum8KQcBY'
               access_secret= 'ot7orENUpLr2RfJV6mWuz0tBrIlIofgT6UGqrUP8RRFds'
               #source code: https://www.youtube.com/watch?v=o_OZdbCzHUA&t=324s
               auth =tweepy.OAuth1UserHandler(twitter_key, twitter_secret)
               auth.set_access_token(access, access_secret)
               
               api= tweepy.API(auth)
               
               def GetCompanyInfo(ticker):
                   return yf.Ticker(ticker).info
               
               if ticker != '':
                   # Get the company information in list format
                   info = GetCompanyInfo(ticker)
                   
                   
               #search for tweets including the company name in the recent week.
               public_tweets= api.search_tweets(info['shortName'])
               list_sentiment=[]
               for tweet in public_tweets: 
                   st.write(tweet.text)
                   
                   analysis = TextBlob(tweet.text)
                   sentiment= analysis.sentiment[0]
                   list_sentiment.append(sentiment)
           #save the mean of the sentiments
           np.mean(list_sentiment)
           
       with col2: 
           #create a loading animation while it retrieves all info
           with st.spinner('Making quantum calculations of sentiment in messages...'):
                time.sleep(5)
                st.success('Done!')
       #Show results with an image depending the messages positivity!
                result= 'Positivity of '+select_newstype+': '+str(round(np.mean(list_sentiment)*100,2))+'%'
                st.header(result)
                if np.mean(list_sentiment)*100> 0:
                    st.image(Image.open('https://github.com/fabriziolufe/financial_programming_streamlit/blob/main/positive.jpg'))
                else: 
                    st.image(Image.open('https://github.com/fabriziolufe/financial_programming_streamlit/blob/main/negative.jpg'))
    
    
#==============================================================================
#Main Body of the App (when running def run the app will start.)
#==============================================================================

def run():
    
    # Add the ticker selection on the sidebar
    # Get the list of stock tickers from S&P500
    ticker_list = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]['Symbol']

    
    #Let's create a sidebar menu, for all 5 tabs that will be present in our Dashboard
    #This idea was inspired in youtube tutorial: https://www.youtube.com/watch?v=hEPoto5xp3k
    
    selected= option_menu(
            menu_title=None,
            options=['Home', 'Summary','Time Analysis','Financials', 'Simulation', 'Sentiment Analysis'],
            icons=['house','book','bar-chart-fill','bank','calculator-fill','twitter'],
            menu_icon='cash-coin',
            orientation='horizontal',
            styles={"nav-link-selected": {"background-color": "indigo"},
                })
    # Add global form
    global ticker
    form_initial= st.sidebar.form(key='Fabrizio')
    form_initial.caption('Which stock would you like to analyze today: ')
    #form including ticker that will update the ticker of analysis to the whole app!
    ticker = form_initial.selectbox("Select a ticker: ", ticker_list)
    #button that will make it happen.
    form_initial.form_submit_button('Update!')
    
    #create a global check box that would include a user option to select a dates range for analysis
    global show_dates
    
    # the check box, it disables when running the twitter sentiment analysis ( since it doesn't depend on dates.)
    if selected == 'Sentiment Analysis':
        
        show_dates = st.sidebar.checkbox("I would like to select a range of dates for my analysis", disabled=True)
    else: 
        show_dates = st.sidebar.checkbox("I would like to select a range of dates for my analysis")
    # Add select begin-end date
    if show_dates:
        form_dates= st.sidebar.form(key='Lucero')
        global start_date, end_date
        col1, col2 = form_dates.columns(2)
        start_date = col1.date_input("Start date", datetime.datetime.today().date() - datetime.timedelta(days=30))
        end_date = col2.date_input("End date", datetime.datetime.today().date())
        #submit button for dates.
        form_dates.form_submit_button('Choose date range')
    #add nice images from yahoo finance and ieseg (for fun)
    
    st.sidebar.caption("In partnership with: ")
    
    image1 = Image.open('https://github.com/fabriziolufe/financial_programming_streamlit/blob/main/Yahoo.png')
    st.sidebar.image(image1)
    image2 = Image.open('https://github.com/fabriziolufe/financial_programming_streamlit/blob/main/ieseg.png')
    st.sidebar.image(image2)
    
    # Show the selected tab when clicked on it.
    if (selected =='Home'): 
        
        st.header('You are analyzing the '+ticker+' stock. Choose any tab to see further analysis')
            
    if (selected == 'Summary') :
        # Run tab 1
        tab1()
    elif (selected == 'Time Analysis') :
        # Run tab 2
        tab2()
    
    elif (selected == 'Financials') : 
        #run tab 3
        tab3()
    
    elif (selected == 'Simulation') :
        #run tab4
        tab4()
    
    elif( selected == 'Sentiment Analysis') :
        #run tab5
        
        tab5()

#Running application normally     
if (__name__ == "__main__" ):
    run()
    
#Big thanks for the python course, Cheers. FL.
