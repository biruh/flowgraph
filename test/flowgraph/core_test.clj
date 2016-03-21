(ns flowgraph.core-test
  (:require [clojure.test :refer :all]
            [clojure.core.matrix :as m]
            [clojure.core.matrix.operators :as ma]
            [flowgraph.core :as fg]))

(defn learn_2 []
  (let [data         (map (fn [x] {:value x :label (* x 2)}) (fg/getRandomVector 100 0.8))
        g            (fg/graph    (fg/mul :weights :value))
        evaluator    (fg/minimize (fg/sqr (fg/sub :label :output)))]
       (fg/iter g evaluator {:weight (fg/getRandomValue -1 1)} data)))



(deftest multiplication-test
  (testing "element multiplication"
    (is (= (:output (fg/predict (fg/graph (fg/mul :weights :value)) {} {:weights 2 :value 2})) 4)))
  (testing "element multiplication solving"
    (let [g (fg/graph (fg/mul :weights :value))
          e (fg/div :value weights)
          s (fg/solve g e {:weights 2} {:value 2})
          p (fg/predict g s {:value 2})]
    (is (= (:output p) 1)))))

(deftest linear-regresssion-test
  (testing "values should be close to 2"
    (let [r (learn_2)]
      (is (and (< 1.9 (:output r))
               (> 2.1 (:output r)))))))
