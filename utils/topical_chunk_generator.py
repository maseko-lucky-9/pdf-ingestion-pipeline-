#!/usr/bin/env python3
"""Generate topically-grouped chunks for meaningful retrieval evaluation.

5 topics x 5 chunks each = 25 relevant chunks, padded with 175 distractor chunks
from synthetic_chunks/chunks_900_tokens.json so total corpus is 200.
"""

import json
import random
from pathlib import Path

random.seed(42)

TOPICS = {
    "photosynthesis": [
        "Photosynthesis is the biochemical process by which plants, algae, and certain bacteria convert light energy, typically from the sun, into chemical energy stored in glucose molecules. Chlorophyll, the green pigment in plant chloroplasts, absorbs photons primarily in the red and blue wavelengths. The overall reaction can be summarized as six molecules of carbon dioxide combining with six molecules of water in the presence of light energy to produce one molecule of glucose and six molecules of oxygen. This process sustains nearly all life on Earth by providing both food and the oxygen content of the atmosphere.",
        "The light-dependent reactions occur in the thylakoid membranes of chloroplasts. Photosystems II and I absorb photons and excite electrons through an electron transport chain. Water molecules are split in a process called photolysis, releasing oxygen as a byproduct and providing electrons to replace those excited at photosystem II. ATP is generated through chemiosmosis as protons flow down their gradient through ATP synthase, and NADPH is produced when electrons reduce NADP+ at photosystem I.",
        "The Calvin cycle, or light-independent reactions, takes place in the stroma of the chloroplast. Carbon dioxide enters through stomata on leaf surfaces and is fixed onto ribulose-1,5-bisphosphate by the enzyme RuBisCO. The resulting six-carbon intermediate quickly splits into two three-carbon molecules of 3-phosphoglycerate. Using ATP and NADPH from the light reactions, these molecules are reduced to glyceraldehyde-3-phosphate, which can be used to synthesize glucose and regenerate RuBP for continued carbon fixation.",
        "C4 photosynthesis is an adaptation that evolved in many tropical grasses and crops such as maize and sugarcane. By spatially separating the initial carbon fixation from the Calvin cycle, C4 plants concentrate CO2 around RuBisCO and minimize photorespiration. CAM photosynthesis, used by succulents and cacti, separates these processes temporally, fixing carbon at night when stomata can open without excessive water loss and releasing it to the Calvin cycle during the day.",
        "Photosynthetic efficiency is limited by several factors, including light intensity, temperature, CO2 concentration, and water availability. At low light levels, the rate of photosynthesis scales linearly with irradiance; at higher intensities, it plateaus as the enzymatic machinery becomes saturated. RuBisCO's dual affinity for both CO2 and O2 causes photorespiration, which can consume up to 25% of fixed carbon in C3 plants under warm conditions, motivating ongoing research into engineering RuBisCO variants with higher specificity for carbon dioxide.",
    ],
    "neural_networks": [
        "An artificial neural network is a computational model composed of layers of interconnected nodes called neurons. Each neuron applies a weighted sum to its inputs, adds a bias term, and passes the result through a nonlinear activation function such as ReLU, sigmoid, or tanh. Stacking multiple layers creates a deep network capable of approximating highly nonlinear functions. The weights and biases are the learnable parameters, typically initialized randomly and adjusted during training to minimize a loss function.",
        "Backpropagation is the algorithm used to train neural networks by efficiently computing gradients of the loss with respect to every weight. It applies the chain rule of calculus from the output layer back to the input layer, caching intermediate activations during the forward pass and reusing them during the backward pass. The resulting gradients are passed to an optimizer such as stochastic gradient descent, Adam, or RMSprop, which updates the weights in a direction that reduces the loss.",
        "Convolutional neural networks, or CNNs, are specialized for grid-like data such as images. They use convolutional layers that apply learnable filters across spatial regions, exploiting translation invariance and local structure. Pooling layers reduce spatial dimensions and provide some invariance to small translations. Modern architectures like ResNet introduce skip connections that enable training of networks hundreds of layers deep by mitigating the vanishing gradient problem during backpropagation.",
        "Recurrent neural networks process sequential data by maintaining a hidden state that is updated at each time step. Vanilla RNNs suffer from vanishing and exploding gradients on long sequences, which led to the development of LSTM and GRU cells with explicit gating mechanisms. Transformer architectures have largely displaced RNNs in natural language tasks by replacing recurrence with self-attention, allowing parallel computation across positions and capturing long-range dependencies more effectively.",
        "Generalization in neural networks is influenced by model capacity, training data, and regularization. Techniques such as dropout, weight decay, early stopping, and data augmentation combat overfitting. Batch normalization and layer normalization stabilize training by normalizing activations within mini-batches or across features. Transfer learning leverages pretrained representations, allowing models trained on large datasets like ImageNet or web-scale text corpora to be fine-tuned on smaller target tasks with dramatically reduced data requirements.",
    ],
    "options_pricing": [
        "An option is a financial derivative that grants the holder the right, but not the obligation, to buy or sell an underlying asset at a predetermined strike price within a specified time frame. A call option confers the right to buy, while a put option confers the right to sell. Options pricing models aim to determine the fair present value of such contracts by accounting for the underlying asset's price, volatility, time to expiration, risk-free interest rate, and any dividends.",
        "The Black-Scholes model, published in 1973 by Fischer Black and Myron Scholes, provides a closed-form solution for European-style options on non-dividend-paying stocks. The key insight is that a portfolio combining the option with a dynamically adjusted position in the underlying asset can be made risk-free, implying that the option's price must grow at the risk-free rate. The resulting partial differential equation admits a solution expressed in terms of the cumulative normal distribution function.",
        "The binomial options pricing model, developed by Cox, Ross, and Rubinstein, discretizes time into a lattice of up and down moves. At each node, the underlying can move up by factor u or down by factor d with risk-neutral probabilities derived from no-arbitrage arguments. The option value is computed by backward induction from the terminal payoffs. Unlike Black-Scholes, the binomial model naturally handles American-style early exercise by taking the maximum of the continuation value and immediate exercise at each node.",
        "Implied volatility is the volatility input that, when fed into Black-Scholes, reproduces the market-observed option price. Because real option prices deviate from the log-normal assumptions of Black-Scholes, implied volatilities vary across strike prices and maturities, producing the well-known volatility smile or skew. Traders use implied volatility surfaces to price exotic options and to hedge portfolios under the SABR or local volatility frameworks.",
        "Monte Carlo methods simulate thousands of price paths for the underlying asset under the risk-neutral measure, compute the discounted payoff along each path, and average the results. This approach is particularly useful for path-dependent options such as Asian, barrier, or lookback options where closed-form solutions are unavailable. Variance reduction techniques including antithetic variates, control variates, and quasi-random sequences improve convergence rates, while least-squares Monte Carlo handles American-style exercise via regression on basis functions.",
    ],
    "climate_change": [
        "Climate change refers to long-term shifts in global temperature and weather patterns driven primarily by human activities since the industrial revolution. The principal mechanism is the enhanced greenhouse effect, in which gases such as carbon dioxide, methane, and nitrous oxide absorb outgoing infrared radiation and re-emit it in all directions, trapping heat in the lower atmosphere. Atmospheric CO2 concentrations have risen from about 280 parts per million in 1750 to over 420 parts per million today, with the rate of increase accelerating over recent decades.",
        "Fossil fuel combustion for electricity generation, transportation, and industrial heat is the dominant source of anthropogenic carbon dioxide emissions. Land-use changes, particularly deforestation in tropical regions, add further CO2 as stored biomass is burned or decomposes, while also reducing the biosphere's capacity to absorb future emissions. Cement production contributes through the calcination reaction that converts limestone to lime, releasing CO2 as a stoichiometric byproduct regardless of the energy source used.",
        "Methane, with a global warming potential approximately 28 times that of CO2 over a 100-year horizon, is released from livestock enteric fermentation, rice paddies, landfills, and fossil fuel extraction infrastructure. Its atmospheric lifetime of around a decade makes methane mitigation a high-leverage strategy for near-term climate action. Satellite-based remote sensing now identifies individual super-emitter sites, enabling targeted leak detection and repair across oil and gas facilities worldwide.",
        "Feedback loops amplify or dampen the climate response to initial forcings. The ice-albedo feedback is particularly potent in polar regions: as sea ice melts, darker ocean surfaces absorb more solar radiation, accelerating further warming and melt. Water vapor, itself a potent greenhouse gas, increases with warmer temperatures, compounding the effect of CO2 increases. Carbon cycle feedbacks, including permafrost thaw and weakening ocean sinks, threaten to transform natural systems from carbon absorbers into net emitters.",
        "Mitigation pathways consistent with the Paris Agreement's 1.5-degree target require rapid decarbonization across all sectors, reaching net-zero emissions globally by mid-century. This entails deploying renewable electricity at scale, electrifying transport and heating, deploying carbon capture for residual industrial emissions, and restoring natural sinks through reforestation and improved soil management. Adaptation measures such as coastal defenses, drought-resilient agriculture, and early warning systems are increasingly necessary given the warming already locked in by historical emissions.",
    ],
    "immune_system": [
        "The adaptive immune system provides antigen-specific defense that complements the rapid but non-specific innate response. Its two main branches are humoral immunity, mediated by B lymphocytes that produce antibodies, and cell-mediated immunity, mediated by T lymphocytes. Unlike the innate system, adaptive responses improve with repeated exposure to the same pathogen, a property called immunological memory that underlies vaccine efficacy and long-term protection after infection.",
        "B cells recognize antigens directly through membrane-bound immunoglobulin receptors. Upon binding a cognate antigen and receiving helper T cell signals, a B cell proliferates and differentiates into plasma cells that secrete soluble antibodies, or into memory B cells that persist for years. Antibodies neutralize pathogens by blocking receptor interactions, opsonize them for phagocytosis, or activate the classical complement pathway that culminates in membrane attack complex formation and pathogen lysis.",
        "T cells recognize antigens as short peptides presented on major histocompatibility complex molecules on antigen-presenting cells. CD8+ cytotoxic T cells kill virally infected or tumor cells displaying foreign peptides on MHC class I, while CD4+ helper T cells orchestrate the broader response by secreting cytokines that activate B cells, macrophages, and other T cells. Regulatory T cells suppress immune responses to self-antigens and excessive inflammation, maintaining tolerance and preventing autoimmunity.",
        "V(D)J recombination generates the enormous diversity of antigen receptors on B and T cells. During lymphocyte development, random assembly of variable, diversity, and joining gene segments creates up to 10^18 unique receptor specificities, ensuring that at least some lymphocytes can recognize any conceivable antigen. Additional diversity arises from somatic hypermutation in germinal centers, where affinity-matured B cells undergo iterative selection for stronger antigen binding during an ongoing response.",
        "Vaccines train the adaptive immune system to recognize pathogens without causing disease. Live attenuated, inactivated, subunit, and mRNA vaccines present antigens through different mechanisms but all aim to generate memory B and T cells that respond rapidly upon subsequent exposure. Adjuvants enhance immunogenicity by activating innate immune sensors, and prime-boost regimens maximize memory formation. The development of mRNA vaccines during the COVID-19 pandemic demonstrated that immunogenic antigens can be manufactured and deployed within months of identifying a new pathogen.",
    ],
}


def main():
    out_dir = Path(__file__).parent.parent / "data" / "synthetic_chunks"
    out_dir.mkdir(parents=True, exist_ok=True)

    existing_file = out_dir / "chunks_900_tokens.json"
    if existing_file.exists():
        with open(existing_file) as f:
            distractor_chunks = json.load(f)
    else:
        distractor_chunks = []

    corpus = []
    idx = 0
    # 5 topics x 5 chunks = 25 topical chunks at indices 0..24
    for topic, texts in TOPICS.items():
        for t in texts:
            corpus.append({"id": idx, "text": t, "topic": topic, "type": "prose"})
            idx += 1

    # Pad with distractors to 200 total
    random.shuffle(distractor_chunks)
    for c in distractor_chunks:
        if idx >= 200:
            break
        corpus.append({"id": idx, "text": c["text"], "topic": "distractor", "type": c.get("type", "prose")})
        idx += 1

    out = out_dir / "topical_chunks.json"
    out.write_text(json.dumps(corpus, indent=2))
    print(f"✅ Wrote {len(corpus)} chunks to {out}")
    print(f"   {sum(1 for c in corpus if c['topic'] != 'distractor')} topical + "
          f"{sum(1 for c in corpus if c['topic'] == 'distractor')} distractors")


if __name__ == "__main__":
    main()
