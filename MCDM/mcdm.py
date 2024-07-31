import functools
import operator
import numpy as np
from MCDM.bwm import calculate_weight
from MCDM.neutrosophic_number import NeutrosophicNumber


class MCDMProblem:
    def __init__(self, criterias_config, alternates, decisions):
        self.criterias = criterias_config.values.keys()
        self.beneficials = [
            criterias_config.values[criteria].beneficial for criteria in self.criterias
        ]
        self.compared2best = {
            criteria: criterias_config.values[criteria].compared2best
            for criteria in self.criterias
        }
        self.compared2worst = {
            criteria: criterias_config.values[criteria].compared2worst
            for criteria in self.criterias
        }

        self.best_criteria = criterias_config.best_criteria
        self.worst_criteria = criterias_config.worst_criteria
        self.alternates = alternates
        self.decisions = self.prepare_decisions(decisions)

    def prepare_decisions(self, decisions):
        decisions = np.array(decisions, dtype=object)
        n, rows, cols = decisions.shape

        for i in range(n):
            for j in range(rows):
                for k in range(cols):
                    decisions[i, j, k] = NeutrosophicNumber(
                        decisions[i, j, k].low,
                        decisions[i, j, k].mid,
                        decisions[i, j, k].high,
                        decisions[i, j, k].T,
                        decisions[i, j, k].I,
                        decisions[i, j, k].F,
                    )
        return decisions

    def calculate_weight(self):
        weights = calculate_weight(
            self.compared2best,
            self.compared2worst,
            self.best_criteria,
            self.worst_criteria,
        )
        return [weights[criteria] for criteria in self.criterias]

    def aggregate_decisions(self):
        num_decisions, num_alternatives, num_criteria = self.decisions.shape

        # Initialize the aggregated_decisions list with Empty Values
        aggregated_decisions = np.empty([num_alternatives, num_criteria], dtype=object)

        for alt_idx in range(num_alternatives):
            for crit_idx in range(num_criteria):
                aggregated_add_val = functools.reduce(
                    operator.add, self.decisions[:, alt_idx, crit_idx]
                )

                aggregated_add_val.low = aggregated_add_val.low / num_decisions
                aggregated_add_val.mid = aggregated_add_val.mid / num_decisions
                aggregated_add_val.high = aggregated_add_val.high / num_decisions

                aggregated_decisions[alt_idx, crit_idx] = aggregated_add_val

        return aggregated_decisions

    def decisisons_to_crisp_values(self, aggregated_results):
        crisp_results = np.array(
            [
                [criteria.de_nutrosophication() for criteria in alternative]
                for alternative in aggregated_results
            ],
            dtype=np.double,
        )
        return crisp_results

    def normalize_matrix(self, matrix):
        rows, cols = matrix.shape

        normalization_factors = np.sqrt(np.sum(matrix**2, axis=0))

        for i in range(rows):
            for j in range(cols):
                matrix[i, j] = matrix[i, j] / normalization_factors[j]

        return matrix

    def calculate_PIS_NIS(self, matrix):
        max_values = np.max(matrix, axis=0)
        min_values = np.min(matrix, axis=0)

        PIS = np.array(
            [
                max_values[idx] if benefit else min_values[idx]
                for idx, benefit in enumerate(self.beneficials)
            ]
        )
        NIS = np.array(
            [
                min_values[idx] if benefit else max_values[idx]
                for idx, benefit in enumerate(self.beneficials)
            ]
        )

        return PIS, NIS

    def calculate_closeness_coefficient(self, matrix, PIS, NIS):
        positive_distance = np.sqrt(np.sum((matrix - PIS) ** 2, axis=1))
        negative_distance = np.sqrt(np.sum((matrix - NIS) ** 2, axis=1))

        return negative_distance / (positive_distance + negative_distance)

    def solve(self):
        # Calculate the weights of the criterias From BWM Method
        weights = self.calculate_weight()

        # Aggregate the decisions
        aggregated_results = self.aggregate_decisions()

        # Convert the decisions to crisp values
        crisp_results = self.decisisons_to_crisp_values(aggregated_results)

        # Normalize the matrix
        normalized_results = self.normalize_matrix(crisp_results)

        # weight the normalized results
        normalized_weighted_results = normalized_results * weights

        # Calculate the PIS and NIS
        PIS, NIS = self.calculate_PIS_NIS(normalized_weighted_results)

        # Calculate the closeness coefficient
        closeness_coefficient = self.calculate_closeness_coefficient(
            normalized_weighted_results, PIS, NIS
        )

        # Rank the alternatives
        ranked_alternatives = np.argsort(closeness_coefficient)[::-1]
        ranked_alternatives = {
            self.alternates[i]: closeness_coefficient[i] for i in ranked_alternatives
        }

        return ranked_alternatives
