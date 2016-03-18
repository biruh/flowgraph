(ns flowgraph.core-test
  (:require [clojure.test :refer :all]
            [flowgraph.core :refer :all]))

(def sample_random_data (m/emap (fn [x] [x (* x 2)]) (getRandomVector 100 0.8)))



(deftest linear-regresssion-test
  (testing "values should be close to 2"
    (let [xor_solver (graph (mul (variable :weights) (input :value)))
          evaluator  (minimize (sqr (sub :target :output )))
          amount     (count sample_random_data)
          final_output (loop [idx 0
                              weight (getRandomValue)
                              o      (predict xor_solver {:weights weight :value (nth (nth sample_random_data idx) 0) })]
                              (if (>= idx amount)
                                o
                                (let [new_o (predict xor_solver {:weights weight :bias bias :xor (nth (nth xor 0)  0)})
                                      nw    (solve   xor_solver evaluator {:weights weight
                                                                           :bias bias
                                                                           :xor (nth (nth xor 0)  0)
                                                                           :target (nth (nth xor 0) 1)
                                                                           :output new_o})]
                                  (recur (inc idx) (:weights nw) (:bias nw) new_o))))]
      (is (and (< 1.9 final_output) (> 2.1 final_output))))))
