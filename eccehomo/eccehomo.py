import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler
from imblearn.combine import SMOTEENN
from bayes_opt import BayesianOptimization
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
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
    scatter(max_columns = 10)
        Creates a pairplot scatter of fixed columns
    barplot
        Create bar plot on all integer and categorical data.
    make_html
        Create html file of all methods called.
    """
    def __init__(self, df, y_name, output_path = ""):
        """
        Parameters
        ----------
        df : pandas DataFrame
            The dataframe to be analized
        numeric : List
            List of all numeric features
        strings : int, optional
            The number of legs the animal (default is 4)
        """
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
            self.output_path = output_path + "/" 
        self.image_paths = []
    
    
    def describe(self, to_print = False):
        """
        Prepare summary statistics for data. Add basic analysis text. 
        Parameters
        ----------
        to_print : Boolean
            Specify if summary statistics want to be printed on console (default is False)
        """
        if to_print:
            print(self.df.describe())
        self.html += "<h2>Summary Statistics</h2>The notebook has " + str(self.df.shape[0]) + " observations and " + str(self.df.shape[1]) + " columns. Having the following names: " + ', '.join(self.df.columns) + ".<br>"
        self.html += self.df.describe().to_html(buf = None, index = False)
    
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
            for name in aggregators:
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
                image_path = "boxplot" + str(counter) + ".png"
                figure.savefig(self.output_path  + image_path)
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
            image_path = "histograms" + str(counter) + ".png"
            figure.savefig(self.output_path + image_path)
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
        image_path =  "correlation.png"
        figure.savefig(self.output_path + image_path)
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
        image_path =  "pairplot.png"
        sns_figure.savefig(self.output_path + image_path)
        plt.clf()
        self.image_paths.append(image_path)
        
    def barplot(self):
        """
        Create bar plot with frequency 
        Parameters
        ----------
        None
        """
        counter = 1
        for integer in self.integers:
            val_count  = self.df[integer].value_counts().rename("Frequency")
            val_count_df = pd.DataFrame(val_count)
            val_count_df['Class'] = val_count.index
            sns_figure = sns.barplot(x='Class', y="Frequency", data=val_count_df)
            sns_figure.set_title("Frequency of " + integer)
            figure = sns_figure.get_figure()
            image_path = "bar" + str(counter) + ".png"
            figure.savefig(self.output_path + image_path)
            plt.clf()
            self.image_paths.append(image_path)
            counter += 1
        if len(self.unique) > 0:
            for unique_variables in self.unique:
                val_count  = self.df[unique_variables].value_counts().rename("Frequency")
                val_count_df = pd.DataFrame(val_count)
                val_count_df['Class'] = val_count.index
                sns_figure = sns.barplot(x='Class', y="Frequency", data=val_count_df)
                sns_figure.set_title("Frequency of " + integer)
                figure = sns_figure.get_figure()
                image_path =  "bar" + str(counter) + ".png"
                figure.savefig(self.output_path + image_path)
                plt.clf()
                self.image_paths.append(image_path)
                counter += 1

    def make_html(self, name):
        """
        Prepare html file to be printed. 
        Parameters
        ----------
        name : int
            name of file
        """
        for images in self.image_paths:
            self.html += '<img src="' + images +'" alt="Not found"><br>'
        self.html = self.html + "</body></html>"
        with open(self.output_path + name +'.html', 'w') as outfile:
            outfile.write(self.html)

#import pandas as pd; import eccehomo; df = pd.read_csv('titanic.csv'); eda = eccehomo.EDA(df)n_names.append(variable)
#eda.describe(); eda.unique_values(); eda.groupby(); eda.boxplot(); eda.histograms(); eda.empty_values(); eda.correlations(); eda.scatter(); eda.barplot(); eda.make_html()

class DataImputation():
    """
    A class to perform data imputation using different methods, such as zero, median, mode, random, KNN, clustering and deleting.

    ...

    Attributes
    ----------
    df : pandas.DataFrame
        A pandas DataFrame with a binomial variable of interest
    y_name : str
        Name of the binomial variable that might be predicted
    
    Methods
    -------
    describe(to_print=False)
        Print summary statistics such as max, min, median, first and third quartile, count and mean of all numeric variables.
    """
    def __init__(self, df, y_name):
        #if taget variable has null values, there is no clear way to impute. Delete the observation.
        self.df = df.dropna(axis = 0, how ='any', subset=[y_name])
        self.y_name = y_name
        self.imputed_values = {}
    
    def delete_observations(self):
        pass

    def easy_imputation(df, variable, method = 'median'):
         """
        Imputes data according to method specified: zero, max, min, mean, median, mode.
        
        Parameters
        ----------
        df: pandas DataFrame
            pandas DataFrame with a column name like variable
        variable: str
            Name of the variable to input data
        method : str
            Specify the method to use for imputation (default is median)

        Returns
        ---------
        df: pandas DataFrame
            DataFrame with null imputed values on variable
        """
        #vector = df[variable]
        #consider adding bfill and ffill for ordered data
        # if method = 'zero':
        # df[variable] = df[variable].fillna(0, axis = 0)
        # elif method = 'max':
        # df[variable] = df[variable].fillna(vector.max(), axis = 0)
        #elif method = 'min':
        #    df[variable] = df[variable].fillna(vector.min(), axis = 0)
        #elif method = 'mean':
        #    df[variable] = df[variable].fillna(vector.mean(), axis = 0)
        #elif method = 'median':
        #    df[variable] = df[variable].fillna(vector.median(), axis = 0)
        #elif method = 'mode':
        #    df[variable] = df[variable].fillna(vector.mode(), axis = 0)
        #return(df)

    def cluster_imputation(df, variable, algo, sample = 30):
        null_values = df[~df.isnull()]
        if df.shape[0] == 0:
            print("Please choose other method. All observations have at least one null value.")
        #else:
            #if algo == 'kmean':
                #perform min max
                #KMeans()
                #KMeans()
            #elif algo == 'gmm':
            #    pass


    def KNN_imputation(self):
        pass

    def user_defined(self, defined_methods = None):
        if defined_methods is not None:
            for variables, method in defined_methods.items():
                pass
        pass

class Outlier:
    """
    A class to handle outliers using Denisty Based Spatial Clustering with Noise (DBSCAN), Isolation forests, boxplot outlier measure and discretization.
    Using a fixed number of max observations to delete
    ...

    Attributes
    ----------
    df : pandas.DataFrame
        A pandas DataFrame with a binomial variable of interest
    y_name : str
        Name of the binomial variable that might be predicted
    
    Methods
    -------
    tukey(max_outliers = 5)
        Performs boxplot outliers measure in all dataset features. And estabishes a threshold of columns with outliers based on masx_outliers.
    isolation_forest(max_outliers = 5)
        Performs isolation forest for outlier detction in all the dataset, moving max_features hyperparameter to fit max_outliers request.
    dbscan(dbscan_val)
        Performs DBSCAN for outlier detection using eps and min_samples.
    discretization(variable, bin = 3)
        Discretize a particular variable, aiming to avoid outliers by grouping data on a particular variable.
    """
    def __init__(self, df, y_name):
        """
        Parameters
        ----------
        df : pandas DataFrame
            The dataframe to be analized
        y_name : string
            Name of the objective variable
        """
        self.df = df
        #make sure you input correct name
        self.y_name = y_name
        self.numeric = list(self.df.dtypes[self.df.dtypes != "object"].index)
    
    def tukey(self, max_outliers = 5):
        """
        Extract outliers based on boxplot method and a fixed number of columns
        
        Parameters
        ----------
        max_outliers : int
            maximum numer of outliers to extract (default is 5)

        Returns
        ---------
        df: pandas DataFrame
            DataFrame without outliers"""
        X_columns = [name for name in self.numeric if name != self.y_name]
        boolean_dict = {}
        for name in X_columns:
            q1 = self.df[name].quantile(0.25)
            q3 = self.df[name].quantile(0.75)
            iqr = q3 - q1
            boolean = (self.df[name] < q1 - 1.5 * iqr) | (self.df[name] > q3 + 1.5 * iqr)
            boolean_dict[name] = boolean
        dataframe = pd.DataFrame(boolean_dict).sum(axis = 1) 
        opt_threshold = len(X_columns) - 1
        for threshold in reversed(range(1, len(X_columns))):
            print(threshold)
            if dataframe[dataframe > threshold].count() > max_outliers:
                break
            else:
                opt_threshold -= 1
        if opt_threshold == 1:
            #warning
            print("Increase the max_outliers parameter (deafult is 5)")
            return(self.df)
        outlier_index = dataframe[dataframe > opt_threshold + 1].index
        return(self.df.loc[~self.df.index.isin(outlier_index)])

    def isolation_forest(self, max_outliers = 5):
        """
        Extract outliers based on isolation forest algorithm and a fixed number of columns
        
        Parameters
        ----------
        max_outliers : int
            maximum numer of outliers to extract (default is 5)

        Returns
        ---------
        df: pandas DataFrame
            DataFrame without outliers"""
        X_columns = [name for name in self.numeric if name != self.y_name]
        isoforest = None
        for num_columns in reversed(range(1, len(X_columns))):
            isoforest = IsolationForest(n_jobs = -1, random_state = 12, max_features = num_columns).fit_predict(self.df[X_columns])
            if np.unique(isoforest, return_counts=True)[1][0] < max_outliers:
                break
        if len(np.where(isoforest == 1)[0]) == self.df.shape[0]:
            print("No outliers found on isolation forest")
            return(self.df)
        return(self.df.iloc[np.where(isoforest == 1)[0], :])

    def dbscan(self, dbscan_val):
        """
        Extract outliers based on DBSCAN
        
        Parameters
        ----------
        dbscan_val : List[float, int]
            List with numer of eps and min samples for the dbscan method

        Returns
        ---------
        df: pandas DataFrame
            DataFrame without outliers if so"""
        if len(dbscan_val) < 2:
            print("dbscan_val has to be an array with two values, eps and min_samples")
        #try automatic implementation
        X_columns = [name for name in self.numeric if name != self.y_name]
        scaled = MinMaxScaler().fit_transform(self.df[X_columns])
        dbscan = DBSCAN(eps = dbscan_val[0], min_samples =dbscan_val[1], n_jobs = -1).fit_predict(scaled)
        print("Found "+ str(len(np.where(dbscan == - 1)[0])) + " outliers")
        if (len(np.where(dbscan == -1)) == 0) or (len(np.where(dbscan == -1)[0]) == self.df.shape[0]):
            print("No outliers found with " + str(dbscan_val))
            return(self.df)
        return(self.df.iloc[np.where(dbscan != - 1)[0], :])

    def discretization(self, variable, bin=3):
        """
        Discretize variable aiming to avoid outlier effects
        
        Parameters
        ----------
        variable : str
            Name of varible to discretize
        bin : int
            Number of expected bins (default is 3)

        Returns
        ---------
        df: pandas DataFrame
            DataFrame with discretized variable"""
        min = self.df[variable].min()
        max = self.df[variable].max()
        min_max = (max - min) / bin
        bins = ([(val + min) * min_max for val in range(bin)] + [max + 1])
        self.df[variable] = print(pd.cut(x = self.df[variable] ,bins = bins, labels = [i for i in range(bin)]))
        return(self.df)   

class Balancing(SMOTEENN):
    def __init__(self, df, y_name, minority_class = 1):
        self.df = df
        self.y_name = y_name 
        self.minority_class = minority_class
        if self.minority_class == 1:
            self.majority_class = 0
        else:
            self.majority_class = 1

    def undersampling(self, size, with_replace = False, return_sample= False):
        majority_sample = self.df[self.y_name == self.majority_class].sample(size, replace = with_replace, random_state = 12)
        minority = self.df[self.y_name == self.minority_class]
        if return_sample:
            return(majority_sample)
        else:
            return(pd.concat([minority, majority_sample], axis = 1))

    def oversampling(self, size , with_replace = True, return_sample = False):
        minority_sample = self.df[self.y_name == self.minority_class].sample(size, replace = with_replace, random_state = 12)
        if return_sample:
            return(minority_sample)
        else:
            return(pd.concat([self.df, minority_sample], axis = 0))
    
    def under_oversampling(self, size = [200, 200], with_replace = [False, True]):
        if len(size) < 2:
            size = size + [size]
        if len(with_replace) < 2:
            with_replace + [with_replace]
        undersampling = self.undersampling(size = size[0], with_replace=with_replace[0], return_sample=True)
        oversampling = self.oversampling(size = size[1], with_replace=with_replace[1], return_sample=True)
        return(pd.concat([undersampling, oversampling], axis = 1))
    
    #def k_opt():
        #creates optimal finding of k
        #pass
    #def clustering_oversampling():
        #cluster over and undersampling
        #pass

class Modeling():
    """
    A class to model Machine Learning Algorithms using Bayesian optimization on random forest, gradient boosting tree and support vector machine and grid search for lohistic regression with l2 regularization.
    ...

    Attributes
    ----------
    X : pandas.DataFrame or numpy array
        A pandas DataFrame scaled and ready to work as input of model no X.
    y_val : pandas Series or  numpy array
        List with target labels.
    algo : str
        Name of the algorithm that want to be implemented: LogisticRegression, SupportVectorMachine, RandomForest, GradientBoostingTree 
    iter: int
        Number of iterations to be performed on Bayesian Optimization.
    
    Methods
    -------
    optimize(max_outliers = 5)
        performs hyperparameter tunning on selected algorithm with n itertions if using Bayesian Optimization
    """
    def __init__(self, X, y_val, algo, iter, scoring):
        """
        Parameters
        ----------
        X : pandas.DataFrame or numpy array
            A pandas DataFrame scaled and ready to work as input of model no X.
        y_val : pandas Series or  numpy array
            List with target labels.
        algo : str
            Name of the algorithm that want to be implemented: LogisticRegression, SupportVectorMachine, RandomForest, GradientBoostingTree 
        iter: int
            Number of iterations to be performed on Bayesian Optimization.
        """
        self.X = X
        self.y_val = y_val 
        self.algo = algo
        self.scoring = scoring
        self.iter = iter
    
    @staticmethod
    def rfc_cv(n_estimators, max_depth, min_samples_split, min_samples_leaf, max_leaf_nodes, data, targets, scoring = 'accuracy'):
        estimator = RandomForestClassifier(
            n_estimators=n_estimators,
            min_samples_split=min_samples_split,
            max_depth = max_depth,
            min_samples_leaf=min_samples_leaf,
            max_leaf_nodes = max_leaf_nodes,
            random_state=12)
        cval = cross_val_score(estimator, data, targets,
                            scoring=scoring, cv=5)
        return(cval.mean())

    @staticmethod
    def gbt_cv(learning_rate, n_estimators, min_samples_split, min_samples_leaf,max_depth, data, targets, scoring = 'accuracy'):
        estimator = GradientBoostingClassifier(
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            min_samples_split = min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_depth = max_depth,
            random_state=12)
        cval = cross_val_score(estimator, data, targets,
                            scoring=scoring, cv=5)
        return(cval.mean())

    @staticmethod
    def svc_cv(C, degree, data, targets, scoring = 'accuracy'):
        estimator = SVC(
            C=C,
            degree = degree,
            kernel = 'poly',
            random_state=12)
        cval = cross_val_score(estimator, data, targets,
                            scoring=scoring, cv=5)
        return(cval.mean())

    def optimize(self):
        """
        Performs bayesian optimization
        
        Parameters
        ----------
        None

        Returns
        ---------
        estimator: scikit-learn estmator
            Scickit learn optimized with chosen hyperparameters"""
        def rfc_crossval(n_estimators, max_depth, min_samples_split, min_samples_leaf, max_leaf_nodes):
            return(self.rfc_cv(
                n_estimators=int(n_estimators),
                max_depth = int(max_depth),
                min_samples_split=int(min_samples_split),
                min_samples_leaf=int(min_samples_leaf),
                max_leaf_nodes = int(max_leaf_nodes),
                data=self.X,
                targets=self.y_val,
                scoring = self.scoring))
        
        def gbt_crossval(learning_rate, n_estimators, min_samples_split, min_samples_leaf,max_depth):
            return(self.gbt_cv(
                learning_rate=float(learning_rate),
                n_estimators=int(n_estimators),
                min_samples_split = int(min_samples_split),
                min_samples_leaf=int(min_samples_leaf),
                max_depth = int(max_depth), 
                data = self.X,
                targets = self.y_val,
                scoring=self.scoring))
        
        def svc_crossval(C, degree):
            return(self.svc_cv(
                C = float(C),
                degree = int(degree),
                data = self.X,
                targets = self.y_val,
                scoring = self.scoring))
        
        if self.algo == 'RandomForest':
            optimizer = BayesianOptimization(
            f= rfc_crossval,
            pbounds={
                "n_estimators": (10, 250),
                "min_samples_split": (2, 25),
                "max_depth": (2, 300),
                "min_samples_leaf": (1, 25),
                "max_leaf_nodes": (2, 25),
            },
            random_state=12, 
            verbose = 2)
            optimizer.maximize(n_iter=self.iter)
            print("Final result:", optimizer.max)
            max_optimizer = optimizer.max['params']
            return(RandomForestClassifier(max_leaf_nodes =int(max_optimizer['max_leaf_nodes']),
                min_samples_split=int(max_optimizer['min_samples_split']), 
                max_depth = int(max_optimizer["max_depth"]), 
                min_samples_leaf=int(max_optimizer['min_samples_leaf']),
                n_estimators = int(max_optimizer['n_estimators']),
                random_state = 12).fit(self.X, self.y_val))
        elif self.algo == "GradientBoostingTree":
            optimizer = BayesianOptimization(
            f= gbt_crossval,
            pbounds={
                "learning_rate": (0.001, 0.2),
                "min_samples_split": (2, 25),
                "max_depth": (2, 300),
                "min_samples_leaf": (1, 25),
                "n_estimators": (10, 300),
            },
            random_state=12, 
            verbose = 2)
            optimizer.maximize(n_iter=self.iter)
            print("Final result:", optimizer.max)
            max_optimizer = optimizer.max['params']
            return(GradientBoostingClassifier(learning_rate =max_optimizer['learning_rate'],
                min_samples_split=int(max_optimizer['min_samples_split']), 
                max_depth = int(max_optimizer["max_depth"]), 
                min_samples_leaf=int(max_optimizer['min_samples_leaf']),
                n_estimators = int(max_optimizer['n_estimators']), 
                random_state=12).fit(self.X, self.y_val))
        elif self.algo == "SupportVectorMachine":
            optimizer = BayesianOptimization(
            f= svc_crossval,
            pbounds={
                "C": (0.001, 0.9999),
                "degree": (2, 4),
            },
            random_state=12, 
            verbose = 2)
            optimizer.maximize(n_iter=self.iter)
            print("Final result:", optimizer.max)
            max_optimizer = optimizer.max['params']
            return(SVC(C = max_optimizer['C'], 
                degree = int(max_optimizer['degree']), 
                kernel = 'poly',
                random_state=12).fit(self.X, self.y_val))
        elif self.algo == "LogisticRegression":
            lr = LogisticRegression(n_jobs = -1, random_state = 12)
            parameters = {"C":[0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1],
            "fit_intercept":[True, False]}
            grid_lr = GridSearchCV(lr, param_grid = parameter, scoring = self.scoring, n_jobs = -1).fit(self.X, self.y_val)
            return(grid_lr.best_estimator_)
        else:
            print("No valid algorithm")