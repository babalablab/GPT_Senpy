from ..utils import Num


class MetricGroup:
    def __init__(
        self,
        labels: dict[str, set[str | Num]],
        preds: dict[str, set[str | Num]],
        label_category: dict[str, list[str]],
    ):
        self.labels = labels
        self.preds = preds
        self.label_category = label_category

        self.recall_dct = {}
        self.precision_dct = {}
        self.num_labels_dct = {}
        self.num_preds_dct = {}

        for c, label in labels.items():
            if c not in label_category:
                continue
            pred = preds[c] if c in preds else set()
            mt = Metrics(label, pred)
            self.recall_dct[c] = mt.recall

        for c, pred in preds.items():
            if c not in label_category:
                continue
            label = labels[c] if c in labels else set()
            mt = Metrics(label, pred)
            self.precision_dct[c] = mt.precision

        for c in label_category:
            label = labels[c] if c in labels else set()
            pred = preds[c] if c in preds else set()
            mt = Metrics(label, pred)
            self.num_labels_dct[c] = mt.num_labels
            self.num_preds_dct[c] = mt.num_preds

        self.recall = (
            sum(self.recall_dct.values()) / len(self.recall_dct.values())
            if self.recall_dct.values()
            else 0.0
        )
        self.precision = (
            sum(self.precision_dct.values()) / len(self.precision_dct.values())
            if self.precision_dct.values()
            else 0.0
        )
        self.f1 = (
            2 * (self.recall * self.precision) / (self.recall + self.precision)
            if self.recall + self.precision
            else 0.0
        )
        self.num_labels = (
            sum(self.num_labels_dct.values()) if self.num_labels_dct.values() else 0
        )
        self.num_preds = (
            sum(self.num_preds_dct.values()) if self.num_preds_dct.values() else 0
        )

    def export_metrics(self) -> dict[str, int | float]:
        return {
            "recall": self.recall,
            "precision": self.precision,
            "f1": self.f1,
            "num_labels": self.num_labels,
            "num_preds": self.num_preds,
        }


class Metrics:
    def __init__(
        self,
        labels: set[str | Num],
        preds: set[str | Num],
    ):
        assert isinstance(labels, set) and all(
            [isinstance(i, str | Num) for i in labels]
        )
        assert isinstance(preds, set) and all([isinstance(i, str | Num) for i in preds])

        self.labels = labels
        self.preds = preds

        self.num_labels: int = len(labels)
        self.num_preds: int = len(preds)

        self.recall: float = self.__get_recall()
        self.precision: float = self.__get_precision()
        self.f1: float = self.__get_f1()

    def __get_recall(
        self,
        labels: set[str | Num] | None = None,
        preds: set[str | Num] | None = None,
    ) -> float:
        labels = self.__get_value_if_none(labels, "labels")
        preds = self.__get_value_if_none(preds, "preds")

        if len(labels) == 0:
            return 0.0

        denominator = len(labels)
        numerator = sum([i in preds for i in labels])
        return numerator / denominator

    def __get_precision(
        self,
        labels: set[str | Num] | None = None,
        preds: set[str | Num] | None = None,
    ) -> float:
        labels = self.__get_value_if_none(labels, "labels")
        preds = self.__get_value_if_none(preds, "preds")

        if len(preds) == 0:
            return 0.0

        denominator = len(preds)
        numerator = sum([i in labels for i in preds])
        return numerator / denominator

    def __get_f1(
        self,
        labels: set[str | Num] | None = None,
        preds: set[str | Num] | None = None,
    ) -> float:
        labels = self.__get_value_if_none(labels, "labels")
        preds = self.__get_value_if_none(preds, "preds")
        recall, precision = self.__get_recall(labels, preds), self.__get_precision(
            labels, preds
        )

        if recall + precision == 0:
            return 0
        return 2 * recall * precision / (recall + precision)

    def __get_value_if_none(
        self, values: set | None = None, name: str | None = None
    ) -> set[str | Num]:
        if values is None and name is not None:
            return getattr(self, name)
        elif values is not None:
            return values
        else:
            raise ValueError("Values and name cannot be None")
