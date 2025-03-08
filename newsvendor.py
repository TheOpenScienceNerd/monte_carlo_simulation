import numpy as np
import pandas as pd
import plotly.graph_objects as go

# type hints
from typing import Optional, Tuple
import numpy.typing as npt

# Constant and default values
DEFAULT_PERIODS = 5 * 4
DEFAULT_ORDER_QUANTITY = 70

# costs and revenue
DEFAULT_PURCHASE_COST = 33.0
DEFAULT_SALE_PRICE = 50.0
DEFAULT_SALVAGE_PRICE = 5.0

# discrete-distribution data
DEFAULT_PATH_DAY_TYPE = './data/nv_day_type.csv'
DEFAULT_PATH_DEMAND = './data/nv_demand_prob.csv'

N_STREAMS = 4
DEFAULT_SEED = 42

class Discrete:
    """
    Discrete distribution: Sample a value with a given observed probability.
    """
    
    def __init__(
        self,
        values: npt.ArrayLike,
        probabilities: npt.ArrayLike,
        random_seed: Optional[int] = None,
    ):
        """
        Discrete distribution

        Params:
        ------
        values: array-like
            list of sample values. Must be of equal length to freq

        freq: array-like
            list of observed frequencies. Must be of equal length to values

        random_seed, int | SeedSequence, optional (default=None)
            A random seed to reproduce samples. If set to none then a unique
            sample is created.
        """
        if len(values) != len(probabilities):
            error_msg = "values and freq arguments must be of equal length"
            raise ValueError(error_msg)

        self.rng = np.random.default_rng(random_seed)
        self.values = np.asarray(values)
        self.probabilities = np.asarray(probabilities)

    def __repr__(self):
        """String representation of class
        """
        return f"Discrete({self.values=}\n{self.probabilities=})"
    
    def sample(self, size: Optional[int] = None) -> float | np.ndarray:
        """
        Sample fron the Discrete distribution

        Params:
        -------
        size: int, optional (default=None)
            Number of samples to return. If integer then
            numpy array returned.
        """
        sample =  self.rng.choice(self.values, p=self.probabilities, size=size)
        return sample

class Experiment:
    def __init__(
        self, 
        periods: Optional[float] = DEFAULT_PERIODS,
        order_quantity: Optional[int] = DEFAULT_ORDER_QUANTITY, 
        purchase_cost: Optional[float] = DEFAULT_PURCHASE_COST,
        sale_price: Optional[float] = DEFAULT_SALE_PRICE,
        salvage_price: Optional[float] = DEFAULT_SALVAGE_PRICE,
        path_data_type: Optional[str] = DEFAULT_PATH_DAY_TYPE,
        path_demand: Optional[str] = DEFAULT_PATH_DEMAND,
        n_streams: Optional[int] = N_STREAMS, 
        main_seed: Optional[int] = DEFAULT_SEED
    ):
        """
        Initialiser for an experiment

        Parameters:
        ----------
        """
        # store parameters
        self.periods = periods
        self.order_quantity = order_quantity
        self.purchase_cost = purchase_cost
        self.sale_price = sale_price
        # in practice validate!
        self.profit_per_unit = sale_price - purchase_cost
        self.salvage_price = salvage_price
        self.recovery_per_unit = purchase_cost - salvage_price
        
        self.path_data_type = path_data_type
        self.path_demand = path_demand

        # load data to use in distributions
        # In practice I recommend you include some data validation! ☺️
        self.day_type = pd.read_csv(path_data_type)
        self.demand_prob = pd.read_csv(path_demand)

        # sampling setup
        self.main_seed = main_seed
        self.n_streams = n_streams
        self.dists = {}
        self.init_sampling()

    def __repr__(self):
        return "Experiment()"
    
    def set_main_seed(self, main_seed):
        """
        Controls the random sampling
        Parameters:
        ----------
        random_number_set: int
            Used to control the set of pseudo random numbers used by
            the distributions in the simulation.
        """
        self.main_seed = main_seed
        self.init_sampling()

    def init_sampling(self):
        """
        Create the distributions used by the model and initialise
        the random seeds of each.
        """
        # produce n non-overlapping streams
        seed_sequence = np.random.SeedSequence(self.main_seed)
        self.seeds = seed_sequence.spawn(self.n_streams)

        # create distribution used to sample day type
        self.dists["day_type"] = Discrete(
            values=self.day_type["day_type"],
            probabilities=self.day_type["prob"],
            random_seed=self.seeds[0]
        )
        
        # create discrete demand distributions...
        for dt, sd in zip(self.demand_prob.columns[1:].tolist(), self.seeds[1:]):
            self.dists[dt] = Discrete(
                values=self.demand_prob["demand"],
                probabilities=self.demand_prob[dt],
                random_seed=sd
            )


def get_sales_revenue(exp: Experiment, demand: npt.ArrayLike) -> np.ndarray:
    """ Calculate the sales revenue for each day in the problem.

    Parameters:
    ----------
    exp: Experiment
        The simulation parameters

    demand: ArrayLike
        The demand where each element represents a day in the simulation.

    Returns:
    --------
    1D numpy array containing sales revenue by day.
    """
    demand = np.asarray(demand)
    
    # default array values to 0 (i.e. no excess stock)
    sales = np.full(len(demand), exp.order_quantity)

    # difference between demand and order quanity
    # positive values means order quantity higher than demand
    diff = exp.order_quantity - demand
    
    sales[diff >= 0] = demand[diff >= 0]
    return sales * exp.sale_price

def get_short_cost(exp: Experiment, demand: npt.ArrayLike) -> np.ndarray:
    """ Calculate the lost sales revenue by day

    Parameters:
    ----------
    exp: Experiment
        The simulation parameters

    demand: ArrayLike
        The demand where each element represents a day in the simulation.

    Returns:
    --------
    1D numpy array containing the lost sales revenue by day.
    """
    demand = np.asarray(demand)
    
    # default array values to 0 (i.e. no shortage)
    shortage = np.zeros(len(demand))

    # difference between demand and order quanity
    # positive values mean demand higher than ordered
    diff = demand - exp.order_quantity
    
    shortage[diff >= 0] = diff[diff >= 0] * exp.profit_per_unit
    return shortage

def get_salvage_value(exp: Experiment, demand: npt.ArrayLike)-> np.ndarray:
    """ Calculate the money salvage by selling excess newspapers

    Parameters:
    ----------
    exp: Experiment
        The simulation parameters

    demand: ArrayLike
        The demand where each element represents a day in the simulation.

    Returns:
    --------
    1D numpy array containing the lost salvaged cost.
    """
    demand = np.asarray(demand)
    
    # default array values to 0 (i.e. no excess stock)
    salvage = np.zeros(len(demand))

    # difference between demand and order quanity
    # positive values means order quantitiy higher than demand
    diff = exp.order_quantity - demand
    
    salvage[diff >= 0] = diff[diff >= 0] * exp.salvage_price
    return salvage

def single_run(exp: Experiment, rep_i: Optional[int]=0)-> pd.DataFrame:
    """ Calculate the lost sales revenue by day

    Parameters:
    ----------
    exp: Experiment
        The simulation parameters

    rep_i: int, optional (default = 0)
        The replication number. Used to set the random seed in the simulation.

    Returns:
    --------
    pandas dataframe. Rows = day in planning period. Cols = results for day.
    """
    # set random seed for the replication
    exp.set_main_seed(rep_i)

    # get the types of day
    day_types = exp.dists["day_type"].sample(exp.periods)

    # get the demand on the day
    demands = np.array([exp.dists[dt].sample() for dt in day_types])

    # daily revenues
    sales_revenue = get_sales_revenue(exp, demands)
    
    # daily shortage costs
    shortage_costs = get_short_cost(exp, demands)

    # daily salvage prices
    salvages = get_salvage_value(exp, demands)

    # cost of purchasing newspapers
    period_costs = np.full(exp.periods, exp.order_quantity * exp.purchase_cost)

    # convert results to pandas dataframe
    period_results = pd.DataFrame()
    period_results['day_type'] = day_types
    period_results['demand'] = demands
    period_results['sales_revenue'] = sales_revenue
    period_results['shortage_cost'] = shortage_costs
    period_results['salvage_revenue'] = salvages
    period_results['period_cost'] = period_costs
    period_results['profit'] = sales_revenue - period_costs - shortage_costs \
        + salvages
    period_results.index += 1 
    period_results.index.name = "day"
    return period_results

def period_statistics(simulated_result: pd.DataFrame):
    """Total profit over the planning period.
    """
    q_stats = {}
    q_stats['profit'] = simulated_result['profit'].sum()
    q_stats['min'] = simulated_result['profit'].min()
    q_stats['max'] = simulated_result['profit'].max()
    q_stats['5th_percentile'] = simulated_result['profit'].quantile(0.05)
    q_stats['95th_percentile'] = simulated_result['profit'].quantile(0.95)
    return q_stats

def multiple_replications(
    experiment: Experiment,
    n_reps: Optional[int] = 5,
) -> np.ndarray:
    """
    Perform multiple replications of the model.

    Params:
    ------
    experiment: Experiment
        The experiment/paramaters to use with model

    n_reps: int, optional (default=5)
        Number of independent replications to run.

    Returns:
    --------
    np.ndarray
    """

    # loop over single run to generate results dicts in a python list.
    results = [period_statistics(single_run(experiment, rep))
               for rep in range(n_reps)]

    # format and return results in a dataframe
    df_results = pd.DataFrame(results)
    df_results.index = np.arange(1, len(df_results) + 1)
    df_results.index.name = "rep"
    return df_results

def create_profit_histogram(profit_data: npt.ArrayLike):

    profit_data = np.asarray(profit_data)
    
    # Calculate statistics
    mean_val = np.mean(profit_data)
    p5 = np.percentile(profit_data, 5)
    p95 = np.percentile(profit_data, 95)
    
    # Create base histogram
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=profit_data,
        name='Profits',
        marker_color='#1f77b4',
        opacity=0.75
    ))
    
    # Add vertical lines with legend entries
    for val, color, name in [
        (mean_val, 'purple', 'Mean'),
        (p5, 'green', '5th Percentile'),
        (p95, 'red', '95th Percentile')
    ]:
        fig.add_vline(
            x=val,
            line_dash='dash',
            line_color=color,
            line_width=2,
            annotation_text=f'{name}: £{val:.2f}',
            annotation_position='top right'
        )
        # Add invisible trace for legend
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='lines',
            line=dict(color=color, dash='dash'),
            name=name
        ))
    
    # Format layout
    fig.update_layout(
        title='Profit Distribution Analysis',
        xaxis_title='Profit (GBP)',
        yaxis_title='Frequency',
        bargap=0.1,
        legend=dict(
            title='Metrics',
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1
        ),
        hovermode='x unified'
    )
    
    return fig


def plot_cumulative_frequency(data):
    # Sort the data
    sorted_data = np.sort(data)
    
    # Calculate cumulative frequency
    y = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    
    # Calculate mean
    mean_value = np.mean(data)
    
    # Create the plot
    fig = go.Figure()
    
    # Add cumulative frequency line
    fig.add_trace(go.Scatter(
        x=sorted_data,
        y=y,
        mode='lines',
        name='Expected Profit',
        line=dict(color='blue')
    ))
    
    # Add mean vertical line
    fig.add_vline(x=mean_value, line_dash="dash", line_color="red",
                  annotation_text="Mean profit",
                  annotation_position="top right")
    
    # Highlight where cumulative distribution crosses 0
    break_even_index = np.searchsorted(sorted_data, 0)
    break_even_y = y[break_even_index]
    
    fig.add_trace(go.Scatter(
        x=[0, 0],
        y=[0, break_even_y],
        mode='lines',
        line=dict(color='green', width=2),
        showlegend=False
    ))
    
    # Add marker at break even point
    fig.add_trace(go.Scatter(
        x=[0],
        y=[break_even_y],
        mode='markers',
        marker=dict(size=10, color='green', symbol='star'),
        name='Break even'
    ))
    
    # Update layout
    fig.update_layout(
        title='Cumulative Frequency Plot',
        xaxis_title='Expected Profit (GBP)',
        yaxis_title='Cumulative Frequency',
        yaxis_range=[0, 1],
        showlegend=True
    )
    
    # Show the plot
    fig.show()
