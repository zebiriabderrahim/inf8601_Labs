#include <assert.h>
#include <stddef.h>

#include "heatsim.h"
#include "log.h"
int heatsim_init(heatsim_t* heatsim, unsigned int dim_x, unsigned int dim_y) {
    int periods[2] = {1, 1};  // Periodic in both dimensions
    int dims[2] = {dim_x, dim_y};

    // Initialize MPI cartesian topology
    MPI_Comm_size(MPI_COMM_WORLD, &heatsim->rank_count);
    MPI_Comm_rank(MPI_COMM_WORLD, &heatsim->rank);

    // Create cartesian communicator
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &heatsim->communicator);

    // Get coordinates in grid
    MPI_Cart_coords(heatsim->communicator, heatsim->rank, 2, heatsim->coordinates);

    // Get neighbor ranks
    MPI_Cart_shift(heatsim->communicator, 0, 1, &heatsim->rank_west_peer,
                   &heatsim->rank_east_peer);
    MPI_Cart_shift(heatsim->communicator, 1, 1, &heatsim->rank_south_peer,
                   &heatsim->rank_north_peer);

    return 0;
}

int heatsim_send_grids(heatsim_t* heatsim, cart2d_t* cart) {
    /*
     * TODO: Envoyer toutes les `grid` aux autres rangs. Cette fonction
     *       est appelé pour le rang 0. Par exemple, si le rang 3 est à la
     *       coordonnée cartésienne (0, 2), alors on y envoit le `grid`
     *       à la position (0, 2) dans `cart`.
     *
     *       Il est recommandé d'envoyer les paramètres `width`, `height`
     *       et `padding` avant les données. De cette manière, le receveur
     *       peut allouer la structure avec `grid_create` directement.
     *
     *       Utilisez `cart2d_get_grid` pour obtenir la `grid` à une coordonnée.
     */

    MPI_Request requests[3 * heatsim->rank_count];
    int request_count = 0;

    for (int y = 0; y < cart->grid_y_count; y++) {
        for (int x = 0; x < cart->grid_x_count; x++) {
            if (y == heatsim->coordinates[1] && x == heatsim->coordinates[0])
                continue;  // Skip self

            grid_t* grid = cart2d_get_grid(cart, x, y);
            int dest_rank;
            MPI_Cart_rank(heatsim->communicator, (int[2]){x, y}, &dest_rank);

            // Send dimensions
            MPI_Isend(&grid->width, 1, MPI_UNSIGNED, dest_rank, 0,
                      heatsim->communicator, &requests[request_count++]);
            MPI_Isend(&grid->height, 1, MPI_UNSIGNED, dest_rank, 1,
                      heatsim->communicator, &requests[request_count++]);

            // Send grid data
            MPI_Isend(grid->data, grid->width * grid->height, MPI_DOUBLE,
                      dest_rank, 2, heatsim->communicator, &requests[request_count++]);
        }
    }

    MPI_Waitall(request_count, requests, MPI_STATUSES_IGNORE);
    return 0;
}

grid_t* heatsim_receive_grid(heatsim_t* heatsim) {
    /*
    * TODO: Recevoir un `grid ` du rang 0. Il est important de noté que
    *       toutes les `grid` ne sont pas nécessairement de la même
    *       dimension (habituellement ±1 en largeur et hauteur). Utilisez
    *       la fonction `grid_create` pour allouer un `grid`.
    *
    *       Utilisez `grid_create` pour allouer le `grid` à retourner.
    */

    unsigned int width, height;

    // Receive dimensions
    MPI_Recv(&width, 1, MPI_UNSIGNED, 0, 0, heatsim->communicator, MPI_STATUS_IGNORE);
    MPI_Recv(&height, 1, MPI_UNSIGNED, 0, 1, heatsim->communicator, MPI_STATUS_IGNORE);

    // Create grid
    grid_t* grid = grid_create(width, height, 0);
    if (!grid) return NULL;

    // Receive data
    MPI_Recv(grid->data, width * height, MPI_DOUBLE, 0, 2,
             heatsim->communicator, MPI_STATUS_IGNORE);

    return grid;
}

int heatsim_exchange_borders(heatsim_t* heatsim, grid_t* grid) {
    assert(grid->padding == 1);

    /*
     * TODO: Échange les bordures de `grid`, excluant le rembourrage, dans le
     *       rembourrage du voisin de ce rang. Par exemple, soit la `grid`
     *       4x4 suivante,
     *
     *                            +-------------+
     *                            | x x x x x x |
     *                            | x A B C D x |
     *                            | x E F G H x |
     *                            | x I J K L x |
     *                            | x M N O P x |
     *                            | x x x x x x |
     *                            +-------------+
     *
     *       où `x` est le rembourrage (padding = 1). Ce rang devrait envoyer
     *
     *        - la bordure [A B C D] au rang nord,
     *        - la bordure [M N O P] au rang sud,
     *        - la bordure [A E I M] au rang ouest et
     *        - la bordure [D H L P] au rang est.
     *
     *       Ce rang devrait aussi recevoir dans son rembourrage
     *
     *        - la bordure [A B C D] du rang sud,
     *        - la bordure [M N O P] du rang nord,
     *        - la bordure [A E I M] du rang est et
     *        - la bordure [D H L P] du rang ouest.
     *
     *       Après l'échange, le `grid` devrait avoir ces données dans son
     *       rembourrage provenant des voisins:
     *
     *                            +-------------+
     *                            | x m n o p x |
     *                            | d A B C D a |
     *                            | h E F G H e |
     *                            | l I J K L i |
     *                            | p M N O P m |
     *                            | x a b c d x |
     *                            +-------------+
     *
     *       Utilisez `grid_get_cell` pour obtenir un pointeur vers une cellule.
     */

    MPI_Datatype row_type, col_type;
    // Create row datatype
    MPI_Type_contiguous(grid->width, MPI_DOUBLE, &row_type);
    MPI_Type_commit(&row_type);

    // Create column datatype
    MPI_Type_vector(grid->height, 1, grid->width_padded, MPI_DOUBLE, &col_type);
    MPI_Type_commit(&col_type);

    // Exchange with north/south neighbors
    MPI_Sendrecv(grid_get_cell(grid, 0, 0), 1, row_type,
                 heatsim->rank_south_peer, 0,
                 grid_get_cell(grid, 0, grid->height), 1, row_type,
                 heatsim->rank_north_peer, 0,
                 heatsim->communicator, MPI_STATUS_IGNORE);

    MPI_Sendrecv(grid_get_cell(grid, 0, grid->height-1), 1, row_type,
                 heatsim->rank_north_peer, 1,
                 grid_get_cell(grid, 0, -1), 1, row_type,
                 heatsim->rank_south_peer, 1,
                 heatsim->communicator, MPI_STATUS_IGNORE);

    // Exchange with east/west neighbors
    MPI_Sendrecv(grid_get_cell(grid, 0, 0), 1, col_type,
                 heatsim->rank_west_peer, 2,
                 grid_get_cell(grid, grid->width, 0), 1, col_type,
                 heatsim->rank_east_peer, 2,
                 heatsim->communicator, MPI_STATUS_IGNORE);

    MPI_Sendrecv(grid_get_cell(grid, grid->width-1, 0), 1, col_type,
                 heatsim->rank_east_peer, 3,
                 grid_get_cell(grid, -1, 0), 1, col_type,
                 heatsim->rank_west_peer, 3,
                 heatsim->communicator, MPI_STATUS_IGNORE);

    MPI_Type_free(&row_type);
    MPI_Type_free(&col_type);

    return 0;
}

int heatsim_send_result(heatsim_t* heatsim, grid_t* grid) {
    assert(grid->padding == 0);

    /*
     * TODO: Envoyer les données (`data`) du `grid` résultant au rang 0. Le
     *       `grid` n'a aucun rembourage (padding = 0);
     */
    return MPI_Send(grid->data, grid->width * grid->height, MPI_DOUBLE,
                    0, 0, heatsim->communicator);
}

int heatsim_receive_results(heatsim_t* heatsim, cart2d_t* cart) {
    /*
     * TODO: Recevoir toutes les `grid` des autres rangs. Aucune `grid`
     *       n'a de rembourage (padding = 0).
     *
     *       Utilisez `cart2d_get_grid` pour obtenir la `grid` à une coordonnée
     *       qui va recevoir le contenue (`data`) d'un autre noeud.
     */
    for (int y = 0; y < cart->grid_y_count; y++) {
        for (int x = 0; x < cart->grid_x_count; x++) {
            if (y == heatsim->coordinates[1] && x == heatsim->coordinates[0])
                continue;  // Skip self

            grid_t* grid = cart2d_get_grid(cart, x, y);
            int src_rank;
            MPI_Cart_rank(heatsim->communicator, (int[2]){x, y}, &src_rank);

            MPI_Recv(grid->data, grid->width * grid->height, MPI_DOUBLE,
                     src_rank, 0, heatsim->communicator, MPI_STATUS_IGNORE);
        }
    }
    return 0;
}
