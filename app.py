from flask import Flask, abort, g, has_request_context, render_template, request, redirect, url_for, session, flash
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio


app = Flask(__name__)
app.secret_key ='secretkey'   #secret key for sessoin management
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False  #to supports a warning from SQLAlchemy
db  = SQLAlchemy(app)  #initialize the database

#user model
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100))
    email = db.Column(db.String(100), unique = True)
    password = db.Column(db.String(100))
   


#Database initialization with app context
with app.app_context():
    db.create_all()  

dataset = pd.read_csv("Enterprise_GenAI_Adoption_Impact.csv")

ANALYSIS_ENDPOINTS = {
    "Executive_overview",
    "GenAI_analysis",
    "global_analysis",
    "workforce_impact_analysis",
    "productivity_analysis",
}
ANALYSIS_FILTER_COLUMNS = {
    "country": "Country",
    "industry": "Industry",
    "tool": "GenAI Tool",
    "year": "Adoption Year",
}


def analysis_filter_state():
    if not has_request_context() or request.endpoint not in ANALYSIS_ENDPOINTS:
        return dataset, {}, {}

    if hasattr(g, "analysis_filter_state"):
        return g.analysis_filter_state

    selected = {
        key: request.args.get(key, "").strip()
        for key in ANALYSIS_FILTER_COLUMNS
    }
    options = {}

    for key, column in ANALYSIS_FILTER_COLUMNS.items():
        candidates = dataset
        for other_key, other_column in ANALYSIS_FILTER_COLUMNS.items():
            value = selected[other_key]
            if other_key != key and value:
                candidates = candidates[candidates[other_column].astype(str) == value]

        values = candidates[column].dropna().astype(str).unique().tolist()
        if selected[key] and selected[key] not in values:
            abort(400, description=f"No matching records for the selected {key} filter.")
        options[key] = sorted(
            values,
            key=lambda value: (not value.isdigit(), int(value) if value.isdigit() else value.casefold()),
        )

    filtered = dataset
    for key, column in ANALYSIS_FILTER_COLUMNS.items():
        if selected[key]:
            filtered = filtered[filtered[column].astype(str) == selected[key]]

    g.analysis_filter_state = (filtered, selected, options)
    return g.analysis_filter_state


def current_analysis_data():
    return analysis_filter_state()[0]


@app.context_processor
def inject_analysis_filters():
    if request.endpoint in ANALYSIS_ENDPOINTS:
        _, selected_filters, filter_options = analysis_filter_state()
        return {
            "analysis_filter_options": filter_options,
            "active_filters": selected_filters,
        }
    return {}

#explore dashboard login requirement
@app.before_request
def require_login_for_analysis():
    if request.endpoint in ANALYSIS_ENDPOINTS:
        if 'user_id' not in session:
            flash('Please login first to explore the dashboard.', 'error')
            return redirect(url_for('login'))

#home page route
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        user = User.query.filter_by(email=email).first()

        if user and check_password_hash(user.password, password):
            session['user_id'] = user.id
            session['user_name'] = user.name
            flash('Login successful!', 'success')
            return redirect(url_for('home'))
        else:
            flash('Invalid email or password.', 'error')
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirm_password']
       

        #validations
        if not name or len(name.strip())<2:
            flash('Name must be at least 2 characters long.', 'error')
            return redirect(url_for('register'))
        
        if not email or '@' not in email:
            flash('Please enter a valid email address.', 'error')
            return redirect(url_for('register'))
        
        #password must be at least 8 characters long and a combination of letters and numbers and special characters
        if len(password)<8 or not any(char.isdigit() for char in password)\
              or not any(char.isalpha() for char in password) or not any(not char.isalnum()\
                                                                          for char in password):
            flash('Password must be at least 8 characters long and contain letters, \
                  numbers, and special characters.', 'error')
            return redirect(url_for('register'))
        
        if password != confirm_password:
            flash('Passwords do not match.', 'error')
            return redirect(url_for('register'))
        
        #check if user already exists
        existing_user = User.query.filter_by(email=email).first()
        if existing_user:
            flash('Email already registered. Please log in.', 'error')
            return redirect(url_for('register'))
        
        #create new user
        hashed_password = generate_password_hash(password)
        new_user = User(
            name=name.strip(),
            email=email.strip(),
            password=hashed_password

        )
        try:
            db.session.add(new_user)
            db.session.commit()
            flash('Registration successful! Please log in.', 'success')
            return redirect(url_for('login'))
        except Exception as e:
            db.session.rollback()
            flash('An error occurred during registration. Please try again.', 'error')
            return redirect(url_for('register'))
        
    return render_template('register.html')


@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out.', 'success')
    return redirect(url_for('home'))

#graphs functions

def tool_adoption(df=None):
    df = current_analysis_data() if df is None else df
    tool_adoption = (
    df["GenAI Tool"]
    .value_counts()
    .reset_index()
)

    tool_adoption.columns = ["GenAI Tool", "Adoption Count"]

    fig1 = px.bar(
        tool_adoption,
        x="GenAI Tool",
        y="Adoption Count",
        text_auto=True,
        title="GenAI Tool Adoption Count"
    )

    fig1.update_layout(
        xaxis_title="GenAI Tool",
        yaxis_title="Number of Records",
        template="plotly_white")
    return pio.to_html(fig1, full_html=False)

#2nd graph
def employees_tool(df=None):
    df = current_analysis_data() if df is None else df
    employees_tool = (
    df.groupby("GenAI Tool")["Number of Employees Impacted"]
    .sum()
    .reset_index()
    .sort_values(
        "Number of Employees Impacted",
        ascending=False
    )
)

    fig2 = px.bar(
        employees_tool,
        x="GenAI Tool",
        y="Number of Employees Impacted",
        text_auto=".2s",
        title="Employees Impacted by GenAI Tool"
    )

    fig2.update_layout(
        xaxis_title="GenAI Tool",
        yaxis_title="Employees Impacted",
        template="plotly_white"
    )
    return pio.to_html(fig2, full_html=False)

#3rd graph
def productivity_tool(df=None):
    df = current_analysis_data() if df is None else df
    productivity_tool = (
    df.groupby("GenAI Tool")["Productivity Change (%)"]
    .mean()
    .reset_index()
    .sort_values(
        "Productivity Change (%)",
        ascending=False
    )
)

    fig3 = px.bar(
        productivity_tool,
        x="GenAI Tool",
        y="Productivity Change (%)",
        text_auto=".2f",
        title="Average Productivity Change by GenAI Tool"
    )

    fig3.update_layout(
        xaxis_title="GenAI Tool",
        yaxis_title="Average Productivity Change (%)",
        template="plotly_white"
   )
    return pio.to_html(fig3, full_html=False)

#4th graph
def training_tool(df=None):
    df = current_analysis_data() if df is None else df
    training_tool = (
    df.groupby("GenAI Tool")["Training Hours Provided"]
    .mean()
    .reset_index()
    .sort_values(
        "Training Hours Provided",
        ascending=False
    )
)

    fig4 = px.bar(
        training_tool,
        x="GenAI Tool",
        y="Training Hours Provided",
        text_auto=".2f",
        title="Average Training Hours by GenAI Tool"
    )

    fig4.update_layout(
        xaxis_title="GenAI Tool",
        yaxis_title="Average Training Hours",
        template="plotly_white"
    )
    return pio.to_html(fig4, full_html=False)

#5th graph

# Tool Adoption Percentage

def tool_adoption_percentage(df=None):
    df = current_analysis_data() if df is None else df

    tool_adoption = (
        df["GenAI Tool"]
        .value_counts()
        .reset_index()
    )

    tool_adoption.columns = ["GenAI Tool", "Adoption Count"]

    fig5 = px.pie(
        tool_adoption,
        names="GenAI Tool",
        values="Adoption Count",
        hole=0.55,
        title="GenAI Tool Adoption Percentage"
    )

    fig5.update_traces(
        textposition="inside",
        textinfo="percent+label"
    )

    fig5.update_layout(template="plotly_white")

    return pio.to_html(fig5, full_html=False)



#6th graph
def training_vs_productivity(df=None):
    df = current_analysis_data() if df is None else df
    tool_scatter = (
    df.groupby("GenAI Tool")
    .agg(
        Avg_Training_Hours=("Training Hours Provided", "mean"),
        Avg_Productivity=("Productivity Change (%)", "mean"),
        Employees_Impacted=("Number of Employees Impacted", "sum")
    )
    .reset_index()
)

    fig6 = px.scatter(
        tool_scatter,
        x="Avg_Training_Hours",
        y="Avg_Productivity",
        size="Employees_Impacted",
        color="GenAI Tool",
        hover_name="GenAI Tool",
        title="Training vs Productivity by GenAI Tool",
        size_max=50
    )

    fig6.update_layout(
        xaxis_title="Average Training Hours",
        yaxis_title="Average Productivity Change (%)",
    template="plotly_white"
    )
    return pio.to_html(fig6, full_html=False)

#graphs and analysis route
@app.route('/GenAI_analysis')
def GenAI_analysis():
    # if 'user_id' not in session:
    #     flash('Please log in to access this page.', 'error')
    #     return redirect(url_for('login'))
    analysis_df = current_analysis_data()
    graph1 = tool_adoption(analysis_df)
    graph2 = employees_tool(analysis_df)
    graph3 = productivity_tool(analysis_df)
    graph4 = training_tool(analysis_df)
    graph5 = tool_adoption_percentage(analysis_df)
    graph6 = training_vs_productivity(analysis_df)
    adoption_counts = analysis_df["GenAI Tool"].value_counts()
    return render_template(
        'GenAI_analysis.html',
        graph1_html=graph1,
        graph2_html=graph2,
        graph3_html=graph3,
        graph4_html=graph4,
        graph5_html=graph5,
        graph6_html=graph6,
        adoption_records=f"{len(analysis_df):,}",
        tools_covered=f"{analysis_df['GenAI Tool'].nunique():,}",
        average_productivity=f"{analysis_df['Productivity Change (%)'].mean():,.1f}%",
        average_training=f"{analysis_df['Training Hours Provided'].mean():,.0f}",
        leading_tool=adoption_counts.idxmax(),
        leading_tool_share=f"{adoption_counts.max() / len(analysis_df):.0%}",
    )


# Executive_overview page graph functions
def industry_adoption(df=None):
    df = current_analysis_data() if df is None else df
    industry_adoption = (
    df.groupby("Industry")["Number of Employees Impacted"]
    .sum()
    .reset_index()
    .sort_values("Number of Employees Impacted", ascending=False)
)

    fig1 = px.bar(
        industry_adoption,
        x="Industry",
        y="Number of Employees Impacted",
        title="GenAI Impact by Industry",
        text_auto=".2s"
    )

    fig1.update_layout(
        xaxis_title="Industry",
        yaxis_title="Employees Impacted",
        template="plotly_white"
    )
    return pio.to_html(fig1, full_html=False)

#2nd graph for Executive_overview
def tool_impact(df=None):
    df = current_analysis_data() if df is None else df
    tool_impact = (
    df.groupby("GenAI Tool")["Number of Employees Impacted"]
    .sum()
    .reset_index()
    .sort_values("Number of Employees Impacted", ascending=False)
)

    fig2 = px.bar(
        tool_impact,
        x="GenAI Tool",
        y="Number of Employees Impacted",
        title="Employees Impacted by GenAI Tool",
        text_auto=".2s"
    )

    fig2.update_layout(
        xaxis_title="GenAI Tool",
        yaxis_title="Employees Impacted",
        template="plotly_white"
    )
    return pio.to_html(fig2, full_html=False)

#3rd graph for Executive_overview
def year_trend(df=None):
    df = current_analysis_data() if df is None else df
    year_trend = (
    df.groupby("Adoption Year")["Number of Employees Impacted"]
    .sum()
    .reset_index()
    .sort_values("Adoption Year")
)

    fig3 = px.line(
        year_trend,
        x="Adoption Year",
        y="Number of Employees Impacted",
        markers=True,
        title="GenAI Adoption Trend Over Years"
    )   

    fig3.update_layout(
        xaxis_title="Adoption Year",
        yaxis_title="Employees Impacted",
        template="plotly_white"
    )
    return pio.to_html(fig3, full_html=False)
    
#4th graph for Executive_overview

def tool_impact_pie(df=None):
    df = current_analysis_data() if df is None else df

    tool_impact = (
        df.groupby("GenAI Tool")["Number of Employees Impacted"]
        .sum()
        .reset_index()
    )

    fig4 = px.pie(
        tool_impact,
        names="GenAI Tool",
        values="Number of Employees Impacted",
        hole=0.55,
        title="Share of Employees Impacted by GenAI Tool"
    )

    fig4.update_traces(
        textposition="inside",
        textinfo="percent+label"
    )

    fig4.update_layout(
        template="plotly_white"
    )

    return pio.to_html(fig4, full_html=False)


#5th graph for Executive_overview
def new_roles_by_industry(df=None):
    df = current_analysis_data() if df is None else df
    industry_roles = (
        df.groupby("Industry")
        ["New Roles Created"]
        .sum()
        .reset_index()
        .sort_values("New Roles Created", ascending=False)
    )

    fig = px.bar(
        industry_roles,
        x="Industry",
        y="New Roles Created",
        text_auto=".2s",
        title="New Roles Created by Industry"
    )

    fig.update_layout(
        xaxis_title="Industry",
        yaxis_title="New Roles Created",
        template="plotly_white"
    )
    return pio.to_html(fig, full_html=False)


#graphs and analysis route
@app.route('/Executive_overview')
def Executive_overview():
    # if 'user_id' not in session:
    #     flash('Please log in to access this page.', 'error')
    #     return redirect(url_for('login'))
    analysis_df = current_analysis_data()
    graph1 = industry_adoption(analysis_df)
    graph2 = tool_impact(analysis_df)
    graph3 = year_trend(analysis_df)
    graph4 = tool_impact_pie(analysis_df)
    graph5 = new_roles_by_industry(analysis_df)
    return render_template(
        'Executive_overview.html',
        graph1_html=graph1,
        graph2_html=graph2,
        graph3_html=graph3,
        graph4_html=graph4,
        graph5_html=graph5,
        company_records=f"{len(analysis_df):,}",
        employees_impacted=f"{analysis_df['Number of Employees Impacted'].sum():,.0f}",
        average_productivity=f"{analysis_df['Productivity Change (%)'].mean():,.1f}%",
        new_roles=f"{analysis_df['New Roles Created'].sum():,.0f}",
    )


#global_analysis page graph functions
def global_impact(df=None):
    df = current_analysis_data() if df is None else df
    country_impact = (
    df.groupby("Country")["Number of Employees Impacted"]
    .sum()
    .reset_index()
    )

    fig1 = px.choropleth(
        country_impact,
        locations="Country",
        locationmode="country names",
        color="Number of Employees Impacted",
        hover_name="Country",
        color_continuous_scale="Blues",
        title="Global GenAI Workforce Impact"
    )

    fig1.update_layout(
        template="plotly_white",
        geo=dict(showframe=False, showcoastlines=True)
    )
    return pio.to_html(fig1, full_html=False)

#2nd graph for global_analysis
def top_countries_impact(df=None):
    df = current_analysis_data() if df is None else df
    country_impact = (
    df.groupby("Country")["Number of Employees Impacted"]
    .sum()
    .reset_index()
)
    top_countries = country_impact.nlargest(
        10, "Number of Employees Impacted"
    ).sort_values("Number of Employees Impacted")

    fig2 = px.bar(
        top_countries,
        x="Number of Employees Impacted",
        y="Country",
        orientation="h",
        text_auto=".2s",
        title="Top 10 Countries by Employees Impacted"
    )

    fig2.update_layout(
        template="plotly_white",
        xaxis_title="Employees Impacted",
        yaxis_title="Country"
    )
    return pio.to_html(fig2, full_html=False)

#3rd graph for global_analysis
def productivity_by_industry(df=None):
    df = current_analysis_data() if df is None else df
    industry_productivity = (
    df.groupby("Industry")["Productivity Change (%)"]
    .mean()
    .reset_index()
    .sort_values("Productivity Change (%)", ascending=False)
)

    fig3 = px.bar(
        industry_productivity,
        x="Industry",
        y="Productivity Change (%)",
        color="Productivity Change (%)",
        color_continuous_scale="Viridis",
        text_auto=".2f",
        title="Average Productivity Change by Industry"
    )

    fig3.update_layout(
        template="plotly_white",
        xaxis_title="Industry",
        yaxis_title="Average Productivity Change (%)"
    )
    return pio.to_html(fig3, full_html=False)

#4th graph for global_analysis
def new_roles_by_country(df=None):
    df = current_analysis_data() if df is None else df
    country_roles = (
        df.groupby("Country")["New Roles Created"]
        .sum()
        .reset_index()
        .nlargest(10, "New Roles Created")
        .sort_values("New Roles Created")
    )

    fig4 = px.bar(
        country_roles,
        x="New Roles Created",
        y="Country",
        orientation="h",
        text_auto=".2s",
        title="Top 10 Countries by New Roles Created"
    )

    fig4.update_layout(
        template="plotly_white",
        xaxis_title="New Roles Created",
        yaxis_title="Country"
    )
    return pio.to_html(fig4, full_html=False)

#5th graph for global_analysis
def country_productivity_vs_impact(df=None):
    df = current_analysis_data() if df is None else df
    country_summary = (
        df.groupby("Country")
        .agg(
            Employees_Impacted=("Number of Employees Impacted", "sum"),
            Avg_Productivity=("Productivity Change (%)", "mean"),
            New_Roles=("New Roles Created", "sum")
        )
        .reset_index()
    )

    fig5 = px.scatter(
        country_summary,
        x="Employees_Impacted",
        y="Avg_Productivity",
        size="New_Roles",
        color="Country",
        hover_name="Country",
        size_max=45,
        title="Country-wise Workforce Impact vs Productivity"
    )

    fig5.update_layout(
        template="plotly_white",
        xaxis_title="Employees Impacted",
        yaxis_title="Average Productivity Change (%)"
    )
    return pio.to_html(fig5, full_html=False)

#6th graph for global_analysis
def industry_tool_treemap(df=None):
    df = current_analysis_data() if df is None else df

    treemap_data = (
        df.groupby(["Industry", "GenAI Tool"])
        ["Number of Employees Impacted"]
        .sum()
        .reset_index()
    )

    fig6 = px.treemap(
        treemap_data,
        path=["Industry", "GenAI Tool"],
        values="Number of Employees Impacted",
        color="Number of Employees Impacted",
        color_continuous_scale="Teal",
        title="Industry and GenAI Tool Distribution"
    )

    fig6.update_layout(template="plotly_white")
    return pio.to_html(fig6, full_html=False)

#7th graph for global_analysis
def country_tool_heatmap(df=None):
    df = current_analysis_data() if df is None else df
    heatmap_data = (
        df.pivot_table(
            index="Country",
            columns="GenAI Tool",
            values="Number of Employees Impacted",
            aggfunc="sum",
            fill_value=0
        )
    )

    top15 = heatmap_data.sum(axis=1).nlargest(15).index
    heatmap_data = heatmap_data.loc[top15]

    fig7 = px.imshow(
        heatmap_data,
        aspect="auto",
        color_continuous_scale="Blues",
        labels={
            "x": "GenAI Tool",
            "y": "Country",
            "color": "Employees Impacted"
        },
        title="Country vs GenAI Tool — Workforce Impact"
    )

    fig7.update_layout(template="plotly_white")
    return pio.to_html(fig7, full_html=False)

#graphs and analysis route
@app.route('/global_analysis')
def global_analysis():
    # if 'user_id' not in session:
    #     flash('Please log in to access this page.', 'error')
    #     return redirect(url_for('login'))
    analysis_df = current_analysis_data()
    graph1 = global_impact(analysis_df)
    graph2 = top_countries_impact(analysis_df)
    graph3 = productivity_by_industry(analysis_df)
    graph4 = new_roles_by_country(analysis_df)
    graph5 = country_productivity_vs_impact(analysis_df)
    graph6 = industry_tool_treemap(analysis_df)
    graph7 = country_tool_heatmap(analysis_df)
    return render_template(
        'global_analysis.html',
        graph1_html=graph1,
        graph2_html=graph2,
        graph3_html=graph3,
        graph4_html=graph4,
        graph5_html=graph5,
        graph6_html=graph6,
        graph7_html=graph7,
        countries_covered=f"{analysis_df['Country'].nunique():,}",
        industries_covered=f"{analysis_df['Industry'].nunique():,}",
        employees_impacted=f"{analysis_df['Number of Employees Impacted'].sum():,.0f}",
        average_productivity=f"{analysis_df['Productivity Change (%)'].mean():,.1f}%",
    )


#workforce_impact_analysis graph function
def emplayees_industry(df=None):
    df = current_analysis_data() if df is None else df
    employees_industry = (
        df.groupby("Industry")["Number of Employees Impacted"]
        .sum()
        .reset_index()
        .sort_values("Number of Employees Impacted", ascending=False)
    )

    fig1 = px.bar(
        employees_industry,
        x="Industry",
        y="Number of Employees Impacted",
        color="Industry",
        text_auto=".2s",
        title="Employees Impacted by Industry"
    )

    fig1.update_layout(
        template="plotly_white",
        showlegend=False,
        xaxis_title="Industry",
        yaxis_title="Employees Impacted"
    )
    return pio.to_html(fig1, full_html=False)

#2nd graph for workforce_impact_analysis
def new_roles_by_industry(df=None):
    df = current_analysis_data() if df is None else df
    roles_industry = (
        df.groupby("Industry")["New Roles Created"]
        .sum()
        .reset_index()
        .sort_values("New Roles Created", ascending=False)
    )

    fig2 = px.bar(
        roles_industry,
        x="Industry",
        y="New Roles Created",
        color="Industry",
        text_auto=".2s",
        title="New Roles Created by Industry"
    )

    fig2.update_layout(
        template="plotly_white",
        showlegend=False,
        xaxis_title="Industry",
        yaxis_title="New Roles Created"
    )
    return pio.to_html(fig2, full_html=False)

# 3rd graph for workforce_impact_analysis
def employees_vs_roles(df=None):
    df = current_analysis_data() if df is None else df

    employees_industry = (
        df.groupby("Industry")["Number of Employees Impacted"]
        .sum()
        .reset_index()
    )

    roles_industry = (
        df.groupby("Industry")["New Roles Created"]
        .sum()
        .reset_index()
    )

    merged_data = employees_industry.merge(
        roles_industry,
        on="Industry"
    )

    fig3 = px.scatter(
        merged_data,
        x="Number of Employees Impacted",
        y="New Roles Created",
        size="Number of Employees Impacted",
        color="Industry",
        hover_name="Industry",
        title="Employees Impacted vs New Roles Created"
    )

    fig3.update_layout(
        template="plotly_white",
        xaxis_title="Total Employees Impacted",
        yaxis_title="Total New Roles Created"
    )

    return pio.to_html(fig3, full_html=False)

#4th graph for workforce_impact_analysis
def workforce_summary(df=None):
    df = current_analysis_data() if df is None else df
    workforce_summary = (
        df.groupby("Industry")
        .agg(
            Employees_Impacted=("Number of Employees Impacted", "sum"),
            New_Roles=("New Roles Created", "sum"),
            Avg_Productivity=("Productivity Change (%)", "mean")
        )
        .reset_index()
    )

    fig4 = px.scatter(
        workforce_summary,
        x="Employees_Impacted",
        y="New_Roles",
        size="Employees_Impacted",
        color="Avg_Productivity",
        hover_name="Industry",
        size_max=55,
        title="Industry Workforce Impact Overview",
        color_continuous_scale="Viridis"
    )

    fig4.update_layout(
        template="plotly_white",
        xaxis_title="Employees Impacted",
        yaxis_title="New Roles Created",
        coloraxis_colorbar_title="Avg Productivity (%)"
    )
    return pio.to_html(fig4, full_html=False)

#5th graph for workforce_impact_analysis
def training_by_industry(df=None):
    df = current_analysis_data() if df is None else df
    training_industry = (
        df.groupby("Industry")["Training Hours Provided"]
        .mean()
        .reset_index()
        .sort_values("Training Hours Provided", ascending=False)
    )

    fig5 = px.bar(
        training_industry,
        x="Industry",
        y="Training Hours Provided",
        color="Industry",
        text_auto=".2f",
        title="Average Training Hours by Industry"
    )

    fig5.update_layout(
        template="plotly_white",
        showlegend=False,
        xaxis_title="Industry",
        yaxis_title="Average Training Hours"
    )
    return pio.to_html(fig5, full_html=False)

#6th graph for workforce_impact_analysis
def training_roles(df=None):
    df = current_analysis_data() if df is None else df
    training_roles = (
        df.groupby("Industry")
        .agg(
            Avg_Training_Hours=("Training Hours Provided", "mean"),
            Total_New_Roles=("New Roles Created", "sum")
        )
        .reset_index()
    )

    fig6 = px.scatter(
        training_roles,
        x="Avg_Training_Hours",
        y="Total_New_Roles",
        color="Industry",
        hover_name="Industry",
        title="Training Hours vs New Roles Created"
    )

    fig6.update_layout(
        template="plotly_white",
        xaxis_title="Average Training Hours",
        yaxis_title="Total New Roles Created"
    )
    return pio.to_html(fig6, full_html=False)


#graphs and analysis route
@app.route('/workforce_impact_analysis')
def workforce_impact_analysis():
    # if 'user_id' not in session:
    #     flash('Please log in to access this page.', 'error')
    #     return redirect(url_for('login'))
    analysis_df = current_analysis_data()
    graph1 = emplayees_industry(analysis_df)
    graph2 = new_roles_by_industry(analysis_df)
    graph3 = employees_vs_roles(analysis_df)
    graph4 = workforce_summary(analysis_df)
    graph5 = training_by_industry(analysis_df)
    graph6 = training_roles(analysis_df)
    industry_impact = analysis_df.groupby("Industry")["Number of Employees Impacted"].sum()
    return render_template(
        'workforce_impact_analysis.html',
        graph1_html=graph1,
        graph2_html=graph2,
        graph3_html=graph3,
        graph4_html=graph4,
        graph5_html=graph5,
        graph6_html=graph6,
        workforce_records=f"{len(analysis_df):,}",
        employees_impacted=f"{analysis_df['Number of Employees Impacted'].sum():,.0f}",
        new_roles=f"{analysis_df['New Roles Created'].sum():,.0f}",
        leading_industry=industry_impact.idxmax(),
    )


#productivity_analysis page graph functions
def productivity_by_industry(df=None):
    df = current_analysis_data() if df is None else df
    productivity_industry = (
        df.groupby("Industry")["Productivity Change (%)"]
        .mean()
        .reset_index()
        .sort_values("Productivity Change (%)", ascending=False)
    )

    fig1 = px.bar(
        productivity_industry,
        x="Industry",
        y="Productivity Change (%)",
        color="Productivity Change (%)",
        color_continuous_scale="Viridis",
        text_auto=".2f",
        title="Average Productivity Change by Industry"
    )

    fig1.update_layout(
        template="plotly_white",
        xaxis_title="Industry",
        yaxis_title="Average Productivity Change (%)"
    )

    return pio.to_html(fig1, full_html=False)

#2nd graph for productivity_analysis
def productivity_by_tool(df=None):
    df = current_analysis_data() if df is None else df
    productivity_tool = (
        df.groupby("GenAI Tool")["Productivity Change (%)"]
        .mean()
        .reset_index()
        .sort_values("Productivity Change (%)", ascending=False)
    )

    fig2 = px.bar(
        productivity_tool,
        x="GenAI Tool",
        y="Productivity Change (%)",
        color="GenAI Tool",
        text_auto=".2f",
        title="Average Productivity Change by GenAI Tool"
    )

    fig2.update_layout(
        template="plotly_white",
        showlegend=False,
        xaxis_title="GenAI Tool",
        yaxis_title="Average Productivity Change (%)"
    )
    return pio.to_html(fig2, full_html=False)

#3rd graph for productivity_analysis
def productivity_trend(df=None):
    df = current_analysis_data() if df is None else df
    year_productivity = (
        df.groupby("Adoption Year", as_index=False)
        ["Productivity Change (%)"].mean()
    )

    fig4 = px.line(
        year_productivity,
        x="Adoption Year",
        y="Productivity Change (%)",
        markers=True,
        title="Productivity Trend Over Adoption Years",
        template="plotly_white",
        text="Productivity Change (%)"
    )

    fig4.update_traces(texttemplate="%{text:.2f}", textposition="top center")
    fig4.update_layout(title_x=0.5, height=500)
    return pio.to_html(fig4, full_html=False)

#4th graph for productivity_analysis
def industry_productivity_treemap(df=None):
    df = current_analysis_data() if df is None else df
    industry_productivity = (
        df.groupby("Industry", as_index=False)
        .agg(
            Average_Productivity=("Productivity Change (%)", "mean"),
            Total_Employees_Impacted=("Number of Employees Impacted", "sum")
        )
    )

    fig5 = px.treemap(
        industry_productivity,
        path=["Industry"],
        values="Total_Employees_Impacted",
        color="Average_Productivity",
        color_continuous_scale="Blues",
        hover_data={
            "Average_Productivity": ":.2f",
            "Total_Employees_Impacted": ":,.0f"
        },
        title="Industry-wise Workforce Impact and Productivity"
    )

    fig5.update_layout(
        template="plotly_white",
        title_x=0.5,
        height=550,
        margin=dict(t=70, l=20, r=20, b=20)
    )
    return pio.to_html(fig5, full_html=False)

#5th graph for productivity_analysis
def training_hours_distribution(df=None):
    df = current_analysis_data() if df is None else df
    fig6 = px.histogram(
        df,
        x="Training Hours Provided",
        nbins=30,
        color_discrete_sequence=["#3977D5"],
        title="Distribution of Training Hours Provided",
        template="plotly_white",
        labels={
            "Training Hours Provided": "Training Hours",
            "count": "Number of Records"
        }
    )

    fig6.update_layout(
        title_x=0.5,
        height=500,
        bargap=0.08,
        xaxis_title="Training Hours Provided",
        yaxis_title="Number of Records"
    )
    return pio.to_html(fig6, full_html=False)

#6th graph for productivity_analysis
def productivity_change_distribution(df=None):
    df = current_analysis_data() if df is None else df
    fig7 = px.box(
        df,
        x="Industry",
        y="Productivity Change (%)",
        color="Industry",
        points=False,
        title="Productivity Change Distribution by Industry"
    )

    fig7.update_layout(
        template="plotly_white",
        showlegend=False,
        xaxis_title="Industry",
        yaxis_title="Productivity Change (%)"
    )
    return pio.to_html(fig7, full_html=False)


#graphs and analysis route
@app.route('/productivity_analysis')
def productivity_analysis():
    # if 'user_id' not in session:
    #     flash('Please log in to access this page.', 'error')
    #     return redirect(url_for('login'))
    analysis_df = current_analysis_data()
    graph1 = productivity_by_industry(analysis_df)
    graph2 = productivity_by_tool(analysis_df)
    graph3 = productivity_trend(analysis_df)
    graph4 = industry_productivity_treemap(analysis_df)
    graph5 = training_hours_distribution(analysis_df)
    graph6 = productivity_change_distribution(analysis_df)
    industry_productivity = analysis_df.groupby("Industry")["Productivity Change (%)"].mean()
    tool_productivity = analysis_df.groupby("GenAI Tool")["Productivity Change (%)"].mean()
    return render_template(
        'productivity_analysis.html',
        graph1_html=graph1,
        graph2_html=graph2,
        graph3_html=graph3,
        graph4_html=graph4,
        graph5_html=graph5,
        graph6_html=graph6,
        productivity_records=f"{len(analysis_df):,}",
        average_productivity=f"{analysis_df['Productivity Change (%)'].mean():,.1f}%",
        leading_industry=industry_productivity.idxmax(),
        leading_tool=tool_productivity.idxmax(),
    )


if __name__=='__main__':
    app.run(debug=True )
