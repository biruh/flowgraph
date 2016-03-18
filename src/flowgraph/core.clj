(ns flowgraph.core
  (:require
    [clojure.core.matrix :as m]
    [clojure.core.matrix.operators :as ma]))

(import '[java.util Random])

(set-current-implementation :vectorz)  ;; use Vectorz as default matrix

(defn ginput [n]
  {:name n :type :input })

(defn gvariable [n]
  {:name n :type :variable })

(defn sample-float [a b]
   (+ a (* (rand) (- b a))))

(defn sample-gaussian [rng mu std]
   (+ mu (* (.nextGaussian rng)  std)))

(defn getRandomVector [ncol std]
  (m/array (repeatedly ncol #(sample-float (* -1 std) std))))

(defn getConstantVector [ncol n]
  (m/array (repeatedly ncol (fn [] n))))

(defn getConstantMatrix [nrow ncol n]
  (m/matrix (repeatedly nrow (fn [] (getConstantVector ncol n)))))

(defn getRandomMatrix [nrow ncol std]
  (let [rng (Random.)]
    (m/matrix
      (repeatedly
        nrow
        (fn []
          (getRandomVector ncol std))))))

(defn tanh [m]
  (m/emap (fn [x] (Math/tanh x)) m))

(defn tanh_d [m] ;grad for z = tanh(x) is (1 - z^2)
 (m/emap (fn [x] (- 1.0 (* x x))) m))

(defn sigmoid [t]
  (/ (+ 1 (Math/exp (- t )))))

(defn sigmoid_d [t]
  (* t (- 1 t )))


(defn sum [& args]
  (let [inputs (map (fn [x]
                           (if (number? x)
                             {:type :constant :output x}
                             (:forward x))) args)
        cinputs (map #(:output %) inputs)
        ob {:inputs inputs
            :gradients []
            :adjusted_inputs []
            :type :sum
            :target 0.0
            :learning_rate 1.0
            :output (apply + (map #(:output %) inputs ))}]
 {:forward ob
  :backward (fn [grad]
              (let [target (+ (:output ob) (* (:output ob) grad))
                    lgrads (* (:learning_rate ob) grad)
                    adinputs (+ cinputs (* lgrads cinputs))
                    adjustedoutput (apply + adinputs)
                    ]
          (assoc ob :grad grad
                 :target target
                 :gradients lgrads
                 :adjusted_inputs adinputs
                 :adjusted_output adjustedoutput)))}))


(defn mul [& args]
  (let [inputs (map (fn [x]
                           (if (number? x)
                             {:type :constant :output x}
                             (:forward x))) args)
        cinputs (map #(:output %) inputs)
        output (apply * (map #(:output %) inputs ))
        ob {:inputs inputs
            :gradients []
            :adjusted_inputs []
            :type :mul
            :target 0.0
            :learning_rate 0.1
            :adjusted_output output
            :output output}]
 {:forward ob
  :backward (fn [grad]
              (let [target (+ (:output ob) (* (:output ob) grad))
                    lgrad (* (:learning_rate ob) grad)
                    grads (m/mul cinputs lgrad)
                    adinputs (m/add grads cinputs)
                    adjustedoutput (apply * adinputs)]
          (assoc ob :grad grad
                    :target target
                    :gradients grads
                    :adjusted_inputs adinputs
                    :adjusted_output adjustedoutput)))}))

(defn toGraphObject [x]
  (cond
    (or (m/matrix? x) (m/vec? x) (number? x))
    {:forward (fn [inputs] {:output x})
     :backward (fn [inputs] inputs)}
    (= (:type x) :input)
    {:forward (fn [inputs]
                (let [k (:name x)]
                  (assert (contains? inputs k))
                  {:output (:value (k inputs))}
                  ))
     :backward (fn [inputs]
                 inputs)}
    (= (:type x) :variable)
    {:forward (fn [inputs]
                (let [k (:name x)]
                  (assert (contains? inputs k))
                  {:output (:value (k inputs))}))
     :backward (fn [inputs]
                (let [k (:name x)
                      value (:value (k inputs))
                      grad (:grad inputs)
                      learning_rate (:learning_rate (k inputs))
                      lgrad (m/mul learning_rate grad)
                      odelta (m/mmul lgrad value )
                      result (m/add value odelta)]
                 (assoc inputs k (assoc (k inputs) :value result))))}
    :else
    x
  ))

(defn mmul [w v]
 (let [wi (toGraphObject w)
       vi (toGraphObject v)
       ob {:type            :mmul
           :adjusted_output []}
       ff (fn [inputs]
               (let [wo ((:forward wi) inputs)
                     vo ((:forward vi) inputs)]
                 (assoc ob
                        :inputs [wo vo]
                        :output (m/mmul (:output vo) (:output wo)))))]
   {:forward  ff
    :backward (fn [inputs]
               ((:backward vi) ((:backward wi) inputs )))}))
