import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def load_data():
    # File uploader
    uploaded_file = st.file_uploader("Upload your stock portfolio Excel file", type=['xlsx', 'xls', 'csv'])
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(('xlsx', 'xls')):
                df = pd.read_excel(uploaded_file)
            else:
                df = pd.read_csv(uploaded_file)
            return df
        except Exception as e:
            st.error(f"Error: {e}")
    return None


def create_sector_analysis(df_filtered):
    st.header("Sector-wise Investment Breakdown")

    # Simple Pie Chart for Sector Distribution
    sector_totals = df_filtered.groupby('Sector Name')['Invested Amount'].sum().reset_index()
    fig_pie = px.pie(
        sector_totals,
        values='Invested Amount',
        names='Sector Name',
        title='How Your Money is Distributed Across Sectors',
        hole=0.4  # Makes it a donut chart, easier to read
    )
    fig_pie.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig_pie)

    # Simple Bar Chart for Sector Performance
    st.subheader("How Each Sector is Performing")
    sector_performance = df_filtered.groupby('Sector Name')['Unrealized P&L %'].mean().reset_index()
    fig_performance = px.bar(
        sector_performance,
        x='Sector Name',
        y='Unrealized P&L %',
        title='Profit/Loss % by Sector',
        color='Unrealized P&L %',
        color_continuous_scale=['red', 'green']  # Red for loss, green for profit
    )
    fig_performance.update_layout(
        xaxis_title="Sector",
        yaxis_title="Profit/Loss %"
    )
    st.plotly_chart(fig_performance)


def create_valuation_analysis(df_filtered):
    st.header("Stock Valuation Overview")

    # Simple Bar Chart for PE Ratios
    avg_pe = df_filtered.groupby('Stock Name')['PE TTM Price to Earnings'].mean().reset_index()
    avg_pe = avg_pe.sort_values('PE TTM Price to Earnings', ascending=True)

    fig_pe = px.bar(
        avg_pe.head(10),  # Show only top 10 for clarity
        x='Stock Name',
        y='PE TTM Price to Earnings',
        title='PE Ratio of Your Stocks (Lower is Generally Better)',
        color='PE TTM Price to Earnings',
        color_continuous_scale='RdYlGn_r'  # Red for high PE, green for low PE
    )
    st.plotly_chart(fig_pe)

    # Simple Scatter Plot for Price vs Value
    fig_value = px.scatter(
        df_filtered,
        x='Price to Book Value',
        y='ROE Annual %',
        title='Price vs Company Performance',
        color='Sector Name',
        hover_data=['Stock Name'],
        labels={
            'Price to Book Value': 'Price to Book (Lower is Better)',
            'ROE Annual %': 'Return on Equity % (Higher is Better)'
        }
    )
    st.plotly_chart(fig_value)


def create_risk_analysis(df_filtered):
    st.header("Risk Assessment")

    # Simple Risk Rating
    df_filtered['Risk_Rating'] = pd.qcut(
        df_filtered['Beta 1Year'],
        q=3,
        labels=['Low Risk', 'Medium Risk', 'High Risk']
    )

    # Count of stocks in each risk category
    risk_counts = df_filtered['Risk_Rating'].value_counts().reset_index()
    fig_risk = px.pie(
        risk_counts,
        values='count',
        names='Risk_Rating',
        title='Risk Distribution of Your Portfolio',
        color='Risk_Rating',
        color_discrete_map={
            'Low Risk': 'green',
            'Medium Risk': 'yellow',
            'High Risk': 'red'
        }
    )
    st.plotly_chart(fig_risk)

    # Simple table of risky stocks
    st.subheader("Stocks to Watch (Higher Risk)")
    high_risk = df_filtered[df_filtered['Risk_Rating'] == 'High Risk'][
        ['Stock Name', 'Beta 1Year', 'Standard Deviation 1Year']
    ].sort_values('Beta 1Year', ascending=False)
    st.dataframe(high_risk)


def create_dividend_analysis(df_filtered):
    st.header("Dividend Income Analysis")

    # Simple bar chart of dividend yields
    top_dividend = df_filtered.nlargest(10, 'Dividend yield 1yr %')
    fig_dividend = px.bar(
        top_dividend,
        x='Stock Name',
        y='Dividend yield 1yr %',
        title='Top 10 Dividend Paying Stocks in Your Portfolio',
        color='Dividend yield 1yr %',
        color_continuous_scale='Greens'
    )
    st.plotly_chart(fig_dividend)


def create_fundamental_analysis(df_filtered):
    st.header("Company Growth & Profit Analysis")

    col1, col2 = st.columns(2)

    with col1:
        # Simple Growth Analysis Bar Chart
        growth_data = df_filtered.nlargest(10, 'Revenue Growth Annual YoY %')
        fig_growth = px.bar(
            growth_data,
            x='Stock Name',
            y='Revenue Growth Annual YoY %',
            title='Top 10 Companies by Revenue Growth',
            color='Revenue Growth Annual YoY %',
            color_continuous_scale=['red', 'green']
        )
        fig_growth.update_layout(
            xaxis_title="Company Name",
            yaxis_title="Revenue Growth %"
        )
        st.plotly_chart(fig_growth)

    with col2:
        # Simple Profit Margin Analysis
        margin_data = df_filtered.nlargest(10, 'Operating Profit Margin Annual %')
        fig_margin = px.bar(
            margin_data,
            x='Stock Name',
            y='Operating Profit Margin Annual %',
            title='Top 10 Companies by Profit Margin',
            color='Operating Profit Margin Annual %',
            color_continuous_scale=['red', 'green']
        )
        fig_margin.update_layout(
            xaxis_title="Company Name",
            yaxis_title="Profit Margin %"
        )
        st.plotly_chart(fig_margin)

    # Sector-wise Average Performance
    st.subheader("How Different Sectors Are Growing")
    sector_metrics = df_filtered.groupby('Sector Name').agg({
        'Operating Profit Margin Annual %': 'mean',
        'Revenue Growth Annual YoY %': 'mean',
    }).reset_index()

    # Simple bar chart for sector performance
    fig_sector = px.bar(
        sector_metrics,
        x='Sector Name',
        y=['Operating Profit Margin Annual %', 'Revenue Growth Annual YoY %'],
        title='Sector Performance Overview',
        barmode='group',
        labels={
            'value': 'Percentage (%)',
            'variable': 'Metric'
        }
    )
    fig_sector.update_layout(
        xaxis_title="Sector",
        yaxis_title="Percentage (%)",
        legend_title="Metrics"
    )

    # Update legend labels to be more user-friendly
    fig_sector.for_each_trace(lambda t: t.update(
        name='Profit Margin' if t.name == 'Operating Profit Margin Annual %' else 'Revenue Growth'
    ))

    st.plotly_chart(fig_sector)


def create_advanced_valuation_analysis(df_filtered):
    st.header("Advanced Valuation Analysis")

    # PEG Ratio Calculation (PE/Growth)
    df_filtered['PEG_Ratio'] = df_filtered['PE TTM Price to Earnings'] / df_filtered['Net Profit TTM Growth %']

    col1, col2 = st.columns(2)

    with col1:
        # Fill NaN values in Market Capitalization with median
        market_cap = df_filtered['Market Capitalization'].fillna(df_filtered['Market Capitalization'].median())
        # Normalize market cap for better visualization
        market_cap_normalized = ((market_cap - market_cap.min()) / (market_cap.max() - market_cap.min()) * 30) + 10

        # PEG Ratio Analysis
        fig_peg = px.scatter(
            df_filtered,
            x='PE TTM Price to Earnings',
            y='Net Profit TTM Growth %',
            color='PEG_Ratio',
            size=market_cap_normalized,  # Use normalized values
            hover_data=['Stock Name'],
            title='PE vs Growth (PEG Analysis)',
            color_continuous_scale='RdYlGn_r'
        )
        st.plotly_chart(fig_peg)

    with col2:
        # Price to Book vs ROE
        fig_pb_roe = px.scatter(
            df_filtered,
            x='Price to Book Value',
            y='ROE Annual %',
            color='Sector Name',
            size=market_cap_normalized,  # Use normalized values
            hover_data=['Stock Name'],
            title='Price to Book vs ROE Analysis'
        )
        st.plotly_chart(fig_pb_roe)


def create_risk_reward_analysis(df_filtered):
    st.header("Risk-Reward Analysis")

    # Calculate Risk-Adjusted Returns
    df_filtered['Sharpe_Ratio'] = (df_filtered['Unrealized P&L %'] - 4.5) / df_filtered['Standard Deviation 1Year']

    # Normalize Invested Amount for size
    invested_amount = df_filtered['Invested Amount'].fillna(df_filtered['Invested Amount'].median())
    invested_amount_normalized = ((invested_amount - invested_amount.min()) /
                                  (invested_amount.max() - invested_amount.min()) * 30) + 10

    # Risk-Reward Quadrant Analysis
    fig_risk_reward = px.scatter(
        df_filtered,
        x='Standard Deviation 1Year',
        y='Unrealized P&L %',
        color='Sharpe_Ratio',
        size=invested_amount_normalized,  # Use normalized values
        hover_data=['Stock Name', 'Sector Name'],
        title='Risk-Reward Quadrant Analysis',
        color_continuous_scale='RdYlGn'
    )

    # Add quadrant lines
    median_risk = df_filtered['Standard Deviation 1Year'].median()
    median_return = df_filtered['Unrealized P&L %'].median()

    fig_risk_reward.add_hline(y=median_return, line_dash="dash", line_color="gray")
    fig_risk_reward.add_vline(x=median_risk, line_dash="dash", line_color="gray")

    # Add quadrant labels
    fig_risk_reward.add_annotation(x=median_risk / 2, y=median_return * 1.5, text="High Return, Low Risk",
                                   showarrow=False)
    fig_risk_reward.add_annotation(x=median_risk * 1.5, y=median_return * 1.5, text="High Return, High Risk",
                                   showarrow=False)
    fig_risk_reward.add_annotation(x=median_risk / 2, y=median_return / 2, text="Low Return, Low Risk", showarrow=False)
    fig_risk_reward.add_annotation(x=median_risk * 1.5, y=median_return / 2, text="Low Return, High Risk",
                                   showarrow=False)

    st.plotly_chart(fig_risk_reward)

    # Risk Metrics Table - Updated with correct column names
    st.subheader("Stock Risk Metrics")
    risk_metrics = df_filtered[['Stock Name', 'Beta 1Year', 'Standard Deviation 1Year', 'Sharpe_Ratio']].sort_values(
        'Sharpe_Ratio', ascending=False)
    risk_metrics.columns = ['Stock Name', 'Beta', 'Volatility (1Y)', 'Sharpe Ratio']  # Rename for display
    st.dataframe(risk_metrics)


def create_investment_recommendations(df_filtered):
    st.header("Investment Insights")

    # Quality Score Calculation
    df_filtered['Quality_Score'] = (
            df_filtered['ROE Annual %'].rank(pct=True) * 0.3 +
            df_filtered['Operating Profit Margin Annual %'].rank(pct=True) * 0.2 +
            df_filtered['Net Profit TTM Growth %'].rank(pct=True) * 0.2 +
            df_filtered['Sharpe_Ratio'].rank(pct=True) * 0.3
    )

    # Value Score Calculation
    df_filtered['Value_Score'] = (
            (1 / df_filtered['PE TTM Price to Earnings']).rank(pct=True) * 0.4 +
            (1 / df_filtered['Price to Book Value']).rank(pct=True) * 0.3 +
            df_filtered['Dividend yield 1yr %'].rank(pct=True) * 0.3
    )

    # Overall Score
    df_filtered['Overall_Score'] = (df_filtered['Quality_Score'] + df_filtered['Value_Score']) / 2

    # Top Investment Opportunities
    st.subheader("Top Investment Opportunities")
    top_opportunities = df_filtered.nlargest(5, 'Overall_Score')[
        ['Stock Name', 'Sector Name', 'ROE Annual %', 'PE TTM Price to Earnings',
         'Price to Book Value', 'Overall_Score']
    ]

    # Rename columns for display
    top_opportunities.columns = [
        'Stock Name', 'Sector', 'ROE %', 'PE Ratio', 'Price to Book', 'Overall Score'
    ]
    st.dataframe(top_opportunities)

    # Investment Insights
    st.subheader("Key Investment Insights")

    # Quality-Value Matrix
    fig_qv = px.scatter(
        df_filtered,
        x='Value_Score',
        y='Quality_Score',
        color='Sector Name',
        size='Invested Amount',  # Changed from Market Capitalization to Invested Amount
        hover_data=['Stock Name'],
        title='Quality-Value Investment Matrix'
    )
    fig_qv.add_hline(y=0.7, line_dash="dash", line_color="green")
    fig_qv.add_vline(x=0.7, line_dash="dash", line_color="green")
    st.plotly_chart(fig_qv)


def create_comprehensive_analysis(df_filtered):
    st.header("Comprehensive Stock Analysis & Recommendations")

    # Calculate additional technical indicators
    df_filtered['RSI_Signal'] = df_filtered['Standard Deviation 1Month'].apply(
        lambda x: 'Oversold' if x > 2 else ('Overbought' if x < 0.5 else 'Neutral')
    )

    # Fundamental Health Score
    df_filtered['Fundamental_Health'] = (
            df_filtered['ROE Annual %'].rank(pct=True) * 0.2 +
            df_filtered['Operating Profit Margin Annual %'].rank(pct=True) * 0.2 +
            df_filtered['Net Profit TTM Growth %'].rank(pct=True) * 0.2 +
            (1 / df_filtered['PE TTM Price to Earnings']).rank(pct=True) * 0.2 +
            df_filtered['Piotroski Score'].rank(pct=True) * 0.2
    )

    # Valuation Score
    df_filtered['Valuation_Score'] = (
            (1 / df_filtered['PE TTM Price to Earnings']).rank(pct=True) * 0.4 +
            (1 / df_filtered['Price to Book Value']).rank(pct=True) * 0.3 +
            (df_filtered['Graham Number'] / df_filtered['Current Price']).rank(pct=True) * 0.3
    )

    # Generate Buy/Sell/Hold Recommendations
    def get_recommendation(row):
        fundamental_score = row['Fundamental_Health']
        valuation_score = row['Valuation_Score']
        momentum = row['Trendlyne Momentum Score']

        if fundamental_score > 0.7 and valuation_score > 0.7:
            return 'Strong Buy'
        elif fundamental_score > 0.6 and valuation_score > 0.6:
            return 'Buy'
        elif fundamental_score < 0.3 and valuation_score < 0.3:
            return 'Sell'
        elif fundamental_score < 0.4 and valuation_score < 0.4:
            return 'Reduce'
        else:
            return 'Hold'

    df_filtered['Recommendation'] = df_filtered.apply(get_recommendation, axis=1)

    # Display Recommendations
    st.subheader("Stock Recommendations")
    recommendations = df_filtered[[
        'Stock Name', 'Sector Name', 'Current Price',
        'Fundamental_Health', 'Valuation_Score', 'Recommendation'
    ]].sort_values('Fundamental_Health', ascending=False)

    # Color code recommendations
    def color_recommendations(val):
        if val == 'Strong Buy':
            return 'background-color: darkgreen; color: white'
        elif val == 'Buy':
            return 'background-color: lightgreen'
        elif val == 'Sell':
            return 'background-color: red; color: white'
        elif val == 'Reduce':
            return 'background-color: salmon'
        return 'background-color: yellow'

    st.dataframe(recommendations.style.applymap(color_recommendations, subset=['Recommendation']))

    # Technical Analysis Indicators
    st.subheader("Technical Analysis Summary")
    col1, col2 = st.columns(2)

    with col1:
        # Momentum Analysis
        momentum_data = df_filtered[[
            'Stock Name', 'Trendlyne Momentum Score',
            'Standard Deviation 1Month', 'RSI_Signal'
        ]].sort_values('Trendlyne Momentum Score', ascending=False)

        fig_momentum = px.bar(
            momentum_data.head(10),
            x='Stock Name',
            y='Trendlyne Momentum Score',
            color='RSI_Signal',
            title='Top 10 Stocks by Momentum',
            color_discrete_map={
                'Oversold': 'green',
                'Neutral': 'yellow',
                'Overbought': 'red'
            }
        )
        st.plotly_chart(fig_momentum)

    with col2:
        # Value vs Growth Matrix
        # Fill NaN values in Market Capitalization with median and normalize
        market_cap = df_filtered['Market Capitalization'].fillna(df_filtered['Market Capitalization'].median())
        market_cap_normalized = ((market_cap - market_cap.min()) / (market_cap.max() - market_cap.min()) * 30) + 10

        fig_vg = px.scatter(
            df_filtered,
            x='PE TTM Price to Earnings',
            y='Net Profit TTM Growth %',
            color='Recommendation',
            size=market_cap_normalized,  # Use normalized values without NaN
            hover_data=['Stock Name'],
            title='Value vs Growth Analysis',
            color_discrete_map={
                'Strong Buy': 'darkgreen',
                'Buy': 'lightgreen',
                'Hold': 'yellow',
                'Reduce': 'salmon',
                'Sell': 'red'
            }
        )

        # Add explanatory annotations
        fig_vg.add_annotation(
            x=df_filtered['PE TTM Price to Earnings'].median(),
            y=df_filtered['Net Profit TTM Growth %'].max(),
            text="Bubble size represents market cap",
            showarrow=False,
            yshift=10
        )

        # Improve layout
        fig_vg.update_layout(
            xaxis_title="PE Ratio (Lower is Better)",
            yaxis_title="Profit Growth % (Higher is Better)"
        )

        st.plotly_chart(fig_vg)

    # Investment Strategy Suggestions
    st.subheader("Investment Strategy Suggestions")

    # Stocks for different strategies
    value_picks = df_filtered[
        (df_filtered['PE TTM Price to Earnings'] < df_filtered['PE TTM Price to Earnings'].median()) &
        (df_filtered['Price to Book Value'] < df_filtered['Price to Book Value'].median()) &
        (df_filtered['ROE Annual %'] > df_filtered['ROE Annual %'].median())
        ]['Stock Name'].tolist()

    growth_picks = df_filtered[
        (df_filtered['Net Profit TTM Growth %'] > df_filtered['Net Profit TTM Growth %'].median()) &
        (df_filtered['Revenue Growth Annual YoY %'] > df_filtered['Revenue Growth Annual YoY %'].median())
        ]['Stock Name'].tolist()

    quality_picks = df_filtered[
        (df_filtered['ROE Annual %'] > df_filtered['ROE Annual %'].median()) &
        (df_filtered['Operating Profit Margin Annual %'] > df_filtered['Operating Profit Margin Annual %'].median()) &
        (df_filtered['Piotroski Score'] >= 7)
        ]['Stock Name'].tolist()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.write("Value Strategy Picks")
        st.write(", ".join(value_picks[:5]))

    with col2:
        st.write("Growth Strategy Picks")
        st.write(", ".join(growth_picks[:5]))

    with col3:
        st.write("Quality Strategy Picks")
        st.write(", ".join(quality_picks[:5]))


def create_portfolio_concentration_analysis(df_filtered):
    st.header("Portfolio Concentration Analysis")

    # Calculate concentration metrics
    total_investment = df_filtered['Invested Amount'].sum()
    df_filtered['Portfolio_Weight'] = (df_filtered['Invested Amount'] / total_investment) * 100

    col1, col2 = st.columns(2)

    with col1:
        # Top Holdings Analysis
        st.subheader("Top Holdings (by Investment)")
        top_holdings = df_filtered.nlargest(5, 'Portfolio_Weight')[
            ['Stock Name', 'Portfolio_Weight', 'Unrealized P&L %']
        ]
        fig_holdings = px.bar(
            top_holdings,
            x='Stock Name',
            y='Portfolio_Weight',
            color='Unrealized P&L %',
            title='Top 5 Holdings Impact',
            color_continuous_scale=['red', 'green']
        )
        fig_holdings.update_layout(
            xaxis_title="Stock",
            yaxis_title="Portfolio Weight (%)"
        )
        st.plotly_chart(fig_holdings)

        # Warning for high concentration
        if top_holdings['Portfolio_Weight'].iloc[0] > 20:
            st.warning(
                f"⚠️ High concentration risk: {top_holdings['Stock Name'].iloc[0]} represents {top_holdings['Portfolio_Weight'].iloc[0]:.1f}% of your portfolio")

    with col2:
        # Sector Concentration
        sector_weights = df_filtered.groupby('Sector Name')['Portfolio_Weight'].sum().sort_values(ascending=False)
        st.subheader("Sector Concentration")
        fig_sector_conc = px.pie(
            values=sector_weights.values,
            names=sector_weights.index,
            title='Sector Weight Distribution'
        )
        st.plotly_chart(fig_sector_conc)

        # Warning for sector concentration
        if sector_weights.iloc[0] > 30:
            st.warning(
                f"⚠️ High sector concentration: {sector_weights.index[0]} represents {sector_weights.iloc[0]:.1f}% of your portfolio")


def create_momentum_strength_analysis(df_filtered):
    st.header("Momentum & Strength Analysis")

    # Calculate momentum scores
    df_filtered['Momentum_Score'] = (
            df_filtered['Trendlyne Momentum Score'].rank(pct=True) * 0.4 +
            df_filtered['Relative returns vs Nifty50 year%'].rank(pct=True) * 0.3 +
            df_filtered['Relative returns vs Sector year%'].rank(pct=True) * 0.3
    )

    col1, col2 = st.columns(2)

    with col1:
        # Relative Performance Analysis
        st.subheader("Stock Performance vs Benchmarks")
        performance_data = df_filtered.nlargest(10, 'Relative returns vs Nifty50 year%')[
            ['Stock Name', 'Relative returns vs Nifty50 year%', 'Relative returns vs Sector year%']
        ]

        fig_rel_perf = px.bar(
            performance_data,
            x='Stock Name',
            y=['Relative returns vs Nifty50 year%', 'Relative returns vs Sector year%'],
            title='Top 10 Stocks - Relative Performance',
            barmode='group'
        )
        st.plotly_chart(fig_rel_perf)

    with col2:
        # Momentum Leaders and Laggards
        st.subheader("Momentum Analysis")
        momentum_data = df_filtered.nlargest(5, 'Momentum_Score')[
            ['Stock Name', 'Trendlyne Momentum Score', 'Momentum_Score']
        ]
        st.write("Momentum Leaders:")
        st.dataframe(momentum_data)

        laggards = df_filtered.nsmallest(5, 'Momentum_Score')[
            ['Stock Name', 'Trendlyne Momentum Score', 'Momentum_Score']
        ]
        st.write("Momentum Laggards:")
        st.dataframe(laggards)


def create_quality_metrics_analysis(df_filtered):
    st.header("Quality Metrics Deep Dive")

    # Calculate quality metrics
    df_filtered['Quality_Composite'] = (
            df_filtered['ROE Annual %'].rank(pct=True) * 0.25 +
            df_filtered['ROCE Annual %'].rank(pct=True) * 0.25 +
            df_filtered['Operating Profit Margin Annual %'].rank(pct=True) * 0.25 +
            df_filtered['Piotroski Score'].rank(pct=True) * 0.25
    )

    col1, col2 = st.columns(2)

    with col1:
        # High Quality Stocks
        st.subheader("Highest Quality Stocks")
        quality_stocks = df_filtered.nlargest(5, 'Quality_Composite')[
            ['Stock Name', 'ROE Annual %', 'ROCE Annual %', 'Piotroski Score', 'Quality_Composite']
        ]
        st.dataframe(quality_stocks)

        # Stocks needing attention
        st.subheader("Stocks Needing Review")
        attention_stocks = df_filtered[
            (df_filtered['ROE Annual %'] < df_filtered['ROE Annual %'].median()) &
            (df_filtered['Operating Profit Margin Annual %'] < df_filtered['Operating Profit Margin Annual %'].median())
            ][['Stock Name', 'ROE Annual %', 'Operating Profit Margin Annual %']]
        st.dataframe(attention_stocks)

    with col2:
        # Quality-Value Matrix
        fig_quality = px.scatter(
            df_filtered,
            x='Price to Book Value',
            y='Quality_Composite',
            color='Sector Name',
            size='Invested Amount',
            hover_data=['Stock Name', 'ROE Annual %'],
            title='Quality vs Value Matrix'
        )
        st.plotly_chart(fig_quality)


def main():
    st.title("Enhanced Stock Portfolio Analytics Dashboard")
    st.write("Upload your stock portfolio to get detailed insights")

    df = load_data()

    if df is not None:
        # Sidebar for filtering
        st.sidebar.header("Filters")
        selected_sector = st.sidebar.multiselect(
            "Select Sectors",
            options=df['Sector Name'].unique(),
            default=df['Sector Name'].unique()
        )

        # Filter dataframe based on selection
        df_filtered = df[df['Sector Name'].isin(selected_sector)]

        # Key Metrics
        st.header("Portfolio Overview")
        col1, col2, col3 = st.columns(3)

        with col1:
            total_investment = df_filtered['Invested Amount'].sum()
            st.metric("Total Investment", f"₹{total_investment:,.2f}")

        with col2:
            current_value = df_filtered['Current Price'].sum()
            st.metric("Current Value", f"₹{current_value:,.2f}")

        with col3:
            total_profit_loss = df_filtered['Unrealized P&L'].sum()
            st.metric("Total P&L", f"₹{total_profit_loss:,.2f}")

        # Add new analysis sections
        create_sector_analysis(df_filtered)
        create_valuation_analysis(df_filtered)
        create_risk_analysis(df_filtered)
        create_dividend_analysis(df_filtered)
        create_fundamental_analysis(df_filtered)
        create_advanced_valuation_analysis(df_filtered)
        create_risk_reward_analysis(df_filtered)
        create_investment_recommendations(df_filtered)
        create_comprehensive_analysis(df_filtered)
        create_portfolio_concentration_analysis(df_filtered)
        create_momentum_strength_analysis(df_filtered)
        create_quality_metrics_analysis(df_filtered)

        # Additional Portfolio Statistics
        st.header("Portfolio Statistics")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Portfolio Beta",
                      f"{df_filtered['Beta 1Year'].mean():.2f}",
                      help="Average Beta of the portfolio")

        with col2:
            st.metric("Avg P/E Ratio",
                      f"{df_filtered['PE TTM Price to Earnings'].mean():.2f}",  # Fixed column name
                      help="Average P/E ratio of the portfolio")

        with col3:
            st.metric("Avg Dividend Yield",
                      f"{df_filtered['Dividend yield 1yr %'].mean():.2f}%",
                      help="Average dividend yield of the portfolio")

        # Show raw data with filters
        st.header("Raw Data Explorer")
        selected_columns = st.multiselect(
            "Select columns to view",
            options=df_filtered.columns.tolist(),
            default=['Stock Name', 'Sector Name', 'Invested Amount', 'Current Price', 'Unrealized P&L %']
        )
        st.dataframe(df_filtered[selected_columns])


if __name__ == "__main__":
    main()