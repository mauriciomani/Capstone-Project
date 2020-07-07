import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pdb

np.random.seed(12)

class EDA():
    """
    A class to perform Exploratory Data Analysis task such as descriptive statistics and ploting. 
    It also creates a html file with all the findings.

    ...

    Attributes
    ----------
    df : pandas.DataFrame
        A pandas DataFrame with a binomial variable of interest
    y_name : str
        Name of the binomial variable that might be predicted
    ouptput_path : str
        path to the file where html and images of plots would be stored (default is "" means no path and store on working path)

    Methods
    -------
    describe(to_print=False)
        Print summary statistics such as max, min, median, first and third quartile, count and mean of all numeric variables.
    unique_values(unique_size = 20)
        Extract the unique values of categorical variables. If more values than unique_size then not used
    groupby(top_variables = None, to_analyse = None, aggregators = None)
        Create aggregate tables over categorical data. For a fixed amount of numerical variables, either choose or randomly assigned
    boxplot(name_string=[])
        Create boxplot for all numeric values by all categorical values separated by objective variable. And export image.
    histograms(y_compare = True)
        Create histograms on all float variables and explote images.
    empty_values(to_print = False)
        Create dataframe with information of null values
    correlations(to_print=False)
        Create a correlations matrix and a heatmap for numeric variables
    scatter
        pass
    make_html
        Create html file of all methods called.
    """
    def __init__(self, df, y_name, output_path = ""):
        self.df = df
        self.numeric = list(self.df.dtypes[self.df.dtypes != "object"].index)
        self.strings = list(self.df.dtypes[self.df.dtypes == "object"].index)
        self.floats = list(self.df.dtypes[self.df.dtypes == "float"].index)
        self.integers = list(self.df.dtypes[self.df.dtypes == "int"].index)
        self.html = """<html><head>Exploratory Data Analysis</head><body>"""
        #unique values of categorical fetures
        self.unique = []
        self.y_name = y_name
        if output_path == '':
            self.output_path = output_path
        else:
            self.output_path = "/" + output_path
        self.image_paths = []
    
    
    def describe(self, to_print = False):
        """
        Prepare summary statistics for data. 
        Parameters
        ----------
        to_print : Boolean
            Specify if summary statistics want to be printed on console (default is False)
        """
        if to_print:
            print(self.df.describe())
        self.html += "<h2>Summary Statistics</h2>" + self.df.describe().to_html(buf = None, index = False)
    
    def unique_values(self, unique_size = 20):
        """"
        Create a table with unique information from Object variables in DataFrame
        Parameters
        ----------
        unique_size = int
            Maximum amount of unique values a variable must have (default is 20)
        """
        unique_values = {}
        if len(self.strings) > 0:
            for variable in self.df[self.strings]:
                unique_value = self.df[variable].unique()
                if len(unique_value) <= unique_size:
                    unique_values[variable] = unique_value
                    self.unique.append(variable)
        if len(unique_values) == 0  & len(self.strings) > 0:
            #use warnings
            print("Increase unique size, found string variables but not printed")
        elif len(self.strings) == 0:
            #use warnings
            print("Function called but no categorical variabls (objects) to analyse for uniqueness")
        else:
            self.html += "<h2>Unique variables</h2>"
            self.html += pd.DataFrame(unique_values).to_html(buf = None, index = False, na_rep = "")
        
    def groupby(self, top_variables = None, to_analyse = None, aggregators = None):
        """Create aggregated tables from categorical variables
        Parameters
        ----------
        top_variables : int
            Max amount of categorical values to check (defautl is None)
        to_analyse = List[str]
            List with variable names to analize (defautl is None)
        aggregators = List[str]
            List with name of variables to aggregate (defautl is None)"""
        if (top_variables is None) and (to_analyse is None):
            numeric_variables = self.numeric
        elif to_analyse is not None:
            numeric_variables = to_analyse
        else:
            numeric_variables = np.random.choice(self.numeric, top_variables)
        if aggregators is not None:
            for name in self.strings:
                self.html += "<h2>" + str(name) + "</h2>\n"
                self.html += self.df.groupby(name)[numeric_variables].mean().to_html(buf = None, index = True)
        elif len(self.strings)>0:
            for name in self.strings:
                self.html += "<h2>" + str(name) + "</h2>\n"
                self.html += self.df.groupby(name)[numeric_variables].mean().to_html(buf = None, index = True)
        else:
            #Use warning instead
            print("Group by function called but no data to group. Pass information to aggregators")

    def boxplot(self, name_string = []):
        """Create boxplot to all catgorical and numerical features taking into account y_name
        Parameters
        ----------
        name_string : List[str]
            List with the name of the categorical variables to use in the x axis. If only one, place within list. (defautl is [])
        """
        numeric = [name for name in self.numeric if name != self.y_name]
        if len(name_string) > 0:
            pass
        elif len(self.unique) > 0:
            name_string = self.unique
        else:
            #Use warning instead
            print("Call unique values function first all add the strings to pass to boxplot")
        counter = 1
        for variable_string in name_string:
            for variable_numeric in numeric:
                sns_figure = sns.boxplot(x = variable_string, y = variable_numeric, data = self.df, hue = self.y_name)
                sns_figure.set_title("Distribution of " + variable_numeric + " by " + variable_string)
                figure = sns_figure.get_figure()
                image_path = self.output_path + "boxplot" + str(counter) + ".png"
                figure.savefig(image_path)
                plt.clf()
                self.image_paths.append(image_path)
                counter += 1

    def histograms(self, y_compare = True):
        """Create histograms on all float values
        Parameters
        ----------
        y_compare : Boolean
            If True then plots two histogramas on same figure, for both objective variable classes (default is True)
        """
        floats = [name for name in self.floats if name != self.y_name]
        class0 = self.df[self.df[self.y_name] == 0]
        class1 = self.df[self.df[self.y_name] == 1]
        counter = 1
        for name in floats:
            if y_compare:
                sns_figure = sns.distplot(class0[name], kde = False, color="orange", label = 'class0', norm_hist = True)
                sns_figure = sns.distplot(class1[name], kde = False, color="blue", label = 'class1', norm_hist = True)
            else:
                sns_figure = sns.distplot(self.df[name], kde = False, color="orange")
            sns_figure.set_title("Histogram of " + name)
            plt.legend()
            figure = sns_figure.get_figure()
            image_path = self.output_path + "histograms" + str(counter) + ".png"
            figure.savefig(image_path)
            plt.clf()
            self.image_paths.append(image_path)
            counter += 1
        
    def empty_values(self, to_print = False):
        """
        Prepare null value statistics. 
        Parameters
        ----------
        to_print : Boolean
            Specify if null values statistics want to be printed on console (default is False)
        """
        null_df = self.df.isnull().sum()
        if to_print:
            print(null_df)
        self.html += "<h2>Null values</h2>" + pd.DataFrame(null_df, index = null_df.index, columns =['Sum_null']).to_html(buf = None, index = True)

    def correlations(self, to_print=False):
        """
        Prepare correlation_matrix. 
        Parameters
        ----------
        to_print : Boolean
            Specify if correlation_matrix want to be printed on console (default is False)
        """
        numeric = [name for name in self.numeric if name != self.y_name]
        correlations = self.df[numeric].corr()
        if to_print:
            print(correlations)
        self.html += "<h2>Correlations matrix</h2>" + correlations.to_html(buf = None, index = True)
        sns_figure = sns.heatmap(correlations, xticklabels=correlations.columns, yticklabels=correlations.columns, cmap = 'seismic')
        sns_figure.set_title("Correlation Heatmap")
        figure = sns_figure.get_figure()
        image_path = self.output_path + "correlation.png"
        figure.savefig(image_path)
        plt.clf()
        self.image_paths.append(image_path)
    
    def scatter(self, max_columns = 10):
        """
        Prepare scatter matrix. To understand binary relations. 
        Parameters
        ----------
        max_columns : int
            Specify the maximum amount of columns to be used in the scatter matrix, if less than total columns, then are chosen randomly. If None then all are plot (default is 10)
        """
        if max_columns is None:
            sns_figure = sns.pairplot(self.df)
        elif len(self.numeric) <= max_columns:
            sns_figure = sns.pairplot(self.df)
        else:
            random_names = np.random.choice(self.numeric, max_columns)
            sns_figure = sns.pairplot(self.df[random_names])
        image_path = self.output_path + "pairplot.png"
        sns_figure.savefig(image_path)
        plt.clf()
        self.image_paths.append(image_path)
        

    def make_html(self):
        """
        Prepare html file to be printed. 
        Parameters
        ----------
        None
        """
        for images in self.image_paths:
            self.html += '<img src="' + images +'" alt="Not found"><br>'
        self.html = self.html + "</body></html>"
        with open(self.output_path + 'summary.html', 'w') as outfile:
            outfile.write(self.html)

#import pandas as pd; import eccehomo; df = pd.read_csv('titanic.csv'); eda = eccehomo.EDA(df)n_names.append(variable)
        